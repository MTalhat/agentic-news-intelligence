"""Hybrid Triage Classifier using Laya SLM with confidence calibration and graceful fallback.

Provides sub-100ms decision classification for RSS articles, outputting
calibrated probabilities, micro-tags, and relevance verification.
"""

import os
import time
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

from taxonomy import (
    MACRO_CATEGORIES,
    MACRO_CRITERIA,
    TAXONOMY_CRITERIA,
    get_subtags_for_macro,
    normalize_macro_tag,
    extract_domain_keywords,
    clean_micro_tags,
)

logger = logging.getLogger("classifier")


@dataclass
class TriageResult:
    macro_tag: str
    macro_tags: List[str]
    micro_tags: List[str]
    confidence: float
    on_topic_score: float
    latency_ms: float
    needs_escalation: bool
    is_off_topic: bool
    probabilities: Dict[str, float] = field(default_factory=dict)


class LayaClassifier:
    """Singleton wrapper for the Laya Decision SLM."""

    _instance = None
    _agent = None
    _init_failed = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LayaClassifier, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        self.conf_threshold = float(os.getenv("LAYA_CONFIDENCE_THRESHOLD", "0.65"))
        self.on_topic_threshold = float(os.getenv("LAYA_ON_TOPIC_THRESHOLD", "0.40"))
        self.model_name = os.getenv("LAYA_MODEL_NAME", "convaiinnovations/laya")

    def _ensure_loaded(self) -> bool:
        if self._agent is not None:
            return True
        if self._init_failed:
            return False

        try:
            # Force 100% offline mode so Hugging Face never checks the internet or prints progress bars
            os.environ["HF_HUB_OFFLINE"] = "1"
            import torch
            import laya

            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Initializing Laya model on device='%s' (offline mode)...", device)
            t0 = time.perf_counter()
            self._agent = laya.load(self.model_name, device=device)
            logger.info("Laya model successfully loaded on %s in %.2fs", self._agent.device, time.perf_counter() - t0)
            return True
        except Exception as e:
            logger.warning("Failed to initialize Laya model: %s. Falling back to LLM-only mode.", e)
            self._init_failed = True
            return False

    def is_available(self) -> bool:
        return self._ensure_loaded()

    def classify(
        self,
        title: str,
        text: str,
        category_hint: Optional[str] = None,
        max_chars: int = 1500,
    ) -> Optional[TriageResult]:
        """Performs sub-100ms taxonomic triage and relevance check on an article.
        
        Returns None if Laya is unavailable so caller can fall back gracefully.
        """
        if not self._ensure_loaded():
            return None

        t0 = time.perf_counter()
        state = {
            "title": title.strip(),
            "body": (text or "").strip()[:max_chars],
        }

        # Step 1: Determine Macro Tags (1-3 categories based on probability distribution)
        macro_conf = 1.0
        macro_probs = {}

        if category_hint:
            primary_macro = normalize_macro_tag(category_hint)
            selected_macros = [primary_macro]
            macro_probs = {primary_macro: 1.0}
        else:
            macro_q = {
                "macro": {
                    "type": "choice",
                    "instructions": "Which domain is this news article primarily about?",
                    "criteria": MACRO_CRITERIA,
                }
            }
            try:
                macro_res = self._agent.predict(state, macro_q)
                ans = macro_res.get("answers", {}).get("macro", {})
                macro_probs = ans.get("probabilities", {})
                sorted_macros = sorted(macro_probs.items(), key=lambda x: x[1], reverse=True)
                
                if sorted_macros:
                    top_cat, top_prob = sorted_macros[0]
                    selected_macros = [normalize_macro_tag(top_cat)]
                    macro_conf = top_prob

                    # Select secondary / tertiary macro tags if probability is significant
                    for cat_name, cat_prob in sorted_macros[1:]:
                        if len(selected_macros) >= 3:
                            break
                        if cat_prob >= 0.28 and cat_prob >= 0.65 * top_prob:
                            norm_cat = normalize_macro_tag(cat_name)
                            if norm_cat not in selected_macros:
                                selected_macros.append(norm_cat)
                else:
                    selected_macros = ["Tech"]
                    macro_conf = 0.5
            except Exception as me:
                logger.warning("Macro classification error: %s", me)
                selected_macros = ["Tech"]
                macro_conf = 0.5

        from taxonomy import refine_macro_classification
        primary_macro, selected_macros = refine_macro_classification(
            title, text or "", selected_macros[0], selected_macros
        )

        # Step 2: Determine Subtags across selected macros & On-Topic relevance
        questions = {
            "on_topic": {
                "type": "noul",
                "instructions": f"Is this article really about {primary_macro}?",
            }
        }
        for m in selected_macros:
            subtags = get_subtags_for_macro(m)
            questions[f"subtag_{m}"] = {
                "type": "choice",
                "instructions": f"Which {m} sub-topic is this news article mainly about?",
                "criteria": subtags,
            }

        try:
            res = self._agent.predict(state, questions)
            latency_ms = (time.perf_counter() - t0) * 1000

            ans = res.get("answers", {})
            on_topic_val = ans.get("on_topic", {}).get("noul", 1.0)
            
            candidate_micro = []
            subtag_confidences = []

            # Step 3: Extract technical nomenclature / domain entities from content FIRST (highest specificity)
            extracted_entities = extract_domain_keywords(title, text or "")
            candidate_micro.extend(extracted_entities)

            # Step 4: Append specialized subtags from selected macros
            for m in selected_macros:
                s_dict = ans.get(f"subtag_{m}", {})
                s_choice = s_dict.get("choice", "")
                s_probs = s_dict.get("probabilities", {})
                if s_probs:
                    subtag_confidences.append(max(s_probs.values()))
                    # Include top subtags with good likelihood
                    for stag, sprob in sorted(s_probs.items(), key=lambda x: x[1], reverse=True):
                        if sprob >= 0.20 or (len(candidate_micro) < 3 and sprob >= 0.10):
                            candidate_micro.append(stag)
                elif s_choice:
                    candidate_micro.append(s_choice)

            # Step 5: Strict Clean - Ban Macro Names, Deduplicate, Target 3-6 tags
            micro_tags = clean_micro_tags(candidate_micro, max_tags=6)

            # Fallback if sparse: pull top criteria from taxonomy
            if len(micro_tags) < 3:
                for m in selected_macros:
                    for k in get_subtags_for_macro(m).keys():
                        if k not in micro_tags:
                            micro_tags.append(k)
                        if len(micro_tags) >= 3:
                            break
                    if len(micro_tags) >= 3:
                        break
            micro_tags = clean_micro_tags(micro_tags, max_tags=6)


            # Overall confidence is driven by primary macro domain classification across the 7 categories
            overall_conf = macro_conf if category_hint is None else 1.0

            is_off_topic = on_topic_val < self.on_topic_threshold
            needs_escalation = overall_conf < self.conf_threshold

            return TriageResult(
                macro_tag=primary_macro,
                macro_tags=selected_macros,
                micro_tags=micro_tags,
                confidence=float(overall_conf),
                on_topic_score=float(on_topic_val),
                latency_ms=latency_ms,
                needs_escalation=needs_escalation,
                is_off_topic=is_off_topic,
                probabilities=macro_probs,
            )
        except Exception as e:
            logger.error("Laya prediction error: %s", e)
            return None

