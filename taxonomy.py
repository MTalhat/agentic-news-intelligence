"""Centralized taxonomy and criteria definitions for hybrid AI triage and enrichment.

Synchronizes macro categories and micro-tags across ingestion, database GIN indexes,
and intent search.
"""

from typing import Dict, List, Optional, Any

# Macro categories officially supported in rss-backend
MACRO_CATEGORIES = [
    "Tech",
    "AI",
    "Hardware",
    "Automotive",
    "Gaming",
    "Finance",
    "Science",
]

# Detailed semantic criteria per macro category for Laya decision model
TAXONOMY_CRITERIA: Dict[str, Dict[str, str]] = {
    "Tech": {
        "software": "software applications, operating systems, cloud computing, developer tools",
        "startups": "startup funding, venture capital, tech acquisitions, tech business",
        "security": "cybersecurity, hacking, vulnerabilities, data privacy, breaches",
        "internet": "social media platforms, web standards, internet censorship, telecom",
        "policy": "tech regulation, antitrust, copyright, government policy on tech",
    },
    "AI": {
        "llm_agents": "large language models, autonomous agents, prompt engineering, chat assistants",
        "models_research": "machine learning research, neural architectures, deep learning papers",
        "ai_chips": "neural processing units, AI accelerators, training clusters, inference hardware",
        "ai_ethics": "AI safety, copyright in training, bias, synthetic media, deepfakes",
        "ai_tools": "consumer AI applications, generative image/audio, AI productivity tools",
    },
    "Hardware": {
        "semiconductors": "microchips, fabrication, TSMC, Intel, AMD, transistor processes",
        "consumer_electronics": "smartphones, laptops, wearable devices, monitors, smart home",
        "gpus": "graphics cards, Nvidia RTX, AMD Radeon, compute boards",
        "storage_memory": "SSDs, RAM, DDR5, memory architecture, data center storage",
    },
    "Automotive": {
        "ev_charging": "electric vehicles, batteries, charging stations, range, charging tech",
        "autonomous": "self-driving technology, FSD, robotaxis, driver assistance, lidar",
        "industry": "carmakers, production lines, auto sales, supply chains, plant closures",
        "reviews": "new vehicle reveals, road tests, specs, comparisons, concept cars",
    },
    "Gaming": {
        "releases": "new video game releases, launch dates, gameplay trailers, game reveals",
        "industry_esports": "game studios, esports tournaments, gaming revenue, layoffs",
        "hardware_consoles": "PlayStation, Xbox, Nintendo Switch, Steam Deck, gaming handhelds",
        "pc_gaming": "Steam, PC game optimization, mods, graphics performance",
    },
    "Finance": {
        "crypto": "cryptocurrency, Bitcoin, Ethereum, DeFi, digital assets, exchanges",
        "markets": "stock markets, equities, trading, ETFs, bonds, earnings reports",
        "macro": "inflation, interest rates, central banks, Federal Reserve, GDP",
        "fintech": "digital payments, banking tech, neo-banks, financial infrastructure",
    },
    "Science": {
        "space": "NASA, SpaceX, astronomy, rockets, space exploration, cosmology",
        "energy": "nuclear fusion, solar power, renewable energy, power grids",
        "biotech": "genetics, vaccines, medicine, pharmaceuticals, biological research",
        "physics": "quantum physics, materials science, nanotechnology",
    },
}

# Macro-level choice criteria for zero-hint classification
MACRO_CRITERIA: Dict[str, str] = {
    "Tech": "general software, web, IT, tech policy, startups, cybersecurity",
    "AI": "artificial intelligence, machine learning, generative models, neural nets, LLMs",
    "Hardware": "chips, semiconductors, consumer gadgets, GPUs, physical electronics",
    "Automotive": "cars, electric vehicles, self-driving tech, automotive manufacturers",
    "Gaming": "video games, gaming consoles, game developers, esports, Steam",
    "Finance": "stocks, cryptocurrency, banking, macroeconomic policy, investing",
    "Science": "space exploration, physics, biotechnology, energy, scientific discovery",
}


BANNED_MICRO_TAGS = {
    "tech", "ai", "hardware", "automotive", "gaming", "finance", "science", "off-topic", "off_topic"
}

# Domain-specific technical keywords and entities to enrich micro-tags
DOMAIN_KEYWORD_RULES = [
    # Semiconductors & Fabrication
    (r"\b2nm\b", "2nm"),
    (r"\b3nm\b", "3nm"),
    (r"\bgaafet\b", "gaafet"),
    (r"\bnanosheet\b", "nanosheet"),
    (r"\btsmc\b", "tsmc"),
    (r"\btransistor\b", "transistors"),
    (r"\bfoundry\b", "foundry"),
    (r"\bwafer\b", "silicon_wafers"),
    # Gaming & Consoles
    (r"\bsteam\s*deck\b", "steam_deck"),
    (r"\boled\b", "oled"),
    (r"\bhandheld\b", "handheld"),
    (r"\bapu\b", "apu"),
    (r"\bvrr\b", "variable_refresh_rate"),
    (r"\bsteamos\b", "steamos"),
    # Batteries & Automotive
    (r"\bsolid[- ]state\b", "solid_state_battery"),
    (r"\bquantumscape\b", "quantumscape"),
    (r"\bporsche\b", "porsche"),
    (r"\bendurance\b", "endurance_testing"),
    (r"\b(?:electric vehicles?|evs?)\b", "electric_vehicles"),
    # AI & Agents
    (r"\bautonomous agent\b", "autonomous_agents"),
    (r"\banthropic\b", "anthropic"),
    (r"\binference\b", "inference_optimization"),
    (r"\bruntime\b", "runtime"),
    (r"\bbenchmarks?\b", "benchmarks"),
    (r"\bnvidia\b", "nvidia"),
    # Cryptography & Security
    (r"\bpost[- ]quantum\b", "post_quantum"),
    (r"\bcryptograph\w+", "cryptography"),
    (r"\bnist\b", "nist_standards"),
    # Space & Science
    (r"\b(?:james webb|jwst)\b", "jwst"),
    (r"\bexoplanet\b", "exoplanet"),
    (r"\bbiosignature\b", "biosignatures"),
    (r"\bk2-18b\b", "k2_18b"),
    # Finance & Central Banking
    (r"\bdigital euro\b", "digital_euro"),
    (r"\b(?:european central bank|ecb)\b", "ecb"),
    (r"\bsettlement\b", "settlement"),
    (r"\binterbank\b", "interbank"),
]


def extract_domain_keywords(title: str, text: str) -> List[str]:
    """Extracts salient domain entities and technical nomenclature from title and text."""
    import re
    combined = f"{title} {text}".lower()
    extracted = []
    for pattern, tag in DOMAIN_KEYWORD_RULES:
        if re.search(pattern, combined):
            if tag not in extracted and tag not in BANNED_MICRO_TAGS:
                extracted.append(tag)
    return extracted


def clean_micro_tags(tags: List[str], max_tags: int = 6) -> List[str]:
    """Filters out macro categories, normalizes to snake_case, deduplicates, and limits length."""
    cleaned = []
    for t in tags:
        norm = t.strip().lower().replace(" ", "_").replace("-", "_")
        if norm and norm not in BANNED_MICRO_TAGS and norm not in cleaned:
            cleaned.append(norm)
    return cleaned[:max_tags]


def get_subtags_for_macro(macro: str) -> Dict[str, str]:
    """Returns subtag dictionary criteria for a given macro tag."""
    # Case-insensitive lookup
    for key, subtags in TAXONOMY_CRITERIA.items():
        if key.lower() == macro.lower():
            return subtags
    # Default to Tech if unknown
    return TAXONOMY_CRITERIA.get("Tech", {})


def normalize_macro_tag(macro: Optional[str]) -> str:
    """Normalizes string to canonical Macro tag capitalization."""
    if not macro:
        return "Tech"
    macro_clean = macro.strip().lower()
    for canonical in MACRO_CATEGORIES:
        if canonical.lower() == macro_clean:
            return canonical
    return "Tech"


# Strict domain-anchor rules for primary macro override and secondary tag validation
_PRIMARY_DOMAIN_ANCHORS = [
    ("Gaming", r"\b(?:steam\s*deck|playstation|ps5|xbox|nintendo|switch\s*2|gamedev|pc\s+gamer|esports|unreal\s+engine|fortnite|tomb\s+raider|warhammer|gameplay|video\s+games?|gaming\s+collection|aaa\s+games?)\b"),
    ("Automotive", r"\b(?:byd|ev\s+charging|electric\s+vehicles?|solid[- ]state\s+batter\w+|full\s+self[- ]driving|robotaxis?|automotive\s+(?:playbook|logistics)|carmakers?|porsche|quantumscape|vehicle\s+control\s+unit|building\s+a\s+car)\b"),
    ("Science", r"\b(?:nasa|james\s+webb|jwst|exoplanet|prima\s+telescope|space\s+telescope|nuclear\s+fusion|fusion\s+reactor|fast\s+radio\s+burst|biosignatures?|terafab)\b"),
    ("Finance", r"\b(?:bitcoin|ethereum|solana|crypto\w*|fintech|stocks?\s+to\s+(?:buy|own)|stock\s+access|jpmorgan|binance|liquidations?|bond\s+yields?|ecb|digital\s+euro|investor\s+pulse|software\s+stocks)\b"),
    ("Hardware", r"\b(?:tsmc|2nm|3nm|gaafet|geforce\s+rtx|rtx\s+spark|radeon|160\s*gb\s+vram|maia\s+300|semiconductors?|wafer|formfactor|teradyne)\b"),
    ("AI", r"\b(?:llm\s+agents?|ai\s+agents?|frontier\s+ai|frontier[- ]class\s+ai|mistral\s+ai|anthropic|claude|openai|prompt\s+engineering|rlhf|minimax|tesla\s+ai\s+demo|recursive\s+self[- ]improvement)\b"),
]

_SECONDARY_TAG_GUARDS = {
    "Gaming": r"\b(?:game\w*|gaming|steam|playstation|ps5|xbox|nintendo|switch|esports|unreal|fortnite|warhammer|dlss|ray\s+tracing|handheld|console)\b",
    "Automotive": r"\b(?:car|cars|vehicle|ev|evs|battery|batteries|tesla|byd|porsche|quantumscape|automotive|driving|robotaxi|motor)\b",
    "Finance": r"\b(?:stock|stocks|market|crypto|bitcoin|ethereum|solana|bank|fintech|invest\w*|profit|earnings|yield|dollar|euro|ecb|sec|valuation|funding)\b",
    "Science": r"\b(?:nasa|space|telescope|webb|jwst|fusion|quantum|physics|biotech|planet|astronomy|scientific|research\s+lab)\b",
}


def refine_macro_classification(
    title: str,
    text: str,
    primary_macro: str,
    selected_macros: Optional[List[str]] = None
) -> tuple[str, List[str]]:
    """
    Applies deterministic domain-anchor guardrails to prevent cross-category contamination
    (e.g., preventing Automotive, EV, or Finance articles from receiving a 'Gaming' tag).
    """
    import re
    title_low = (title or "").lower()
    lead_low = f"{title} {(text or '')[:600]}".lower()

    norm_primary = normalize_macro_tag(primary_macro)
    macros = [normalize_macro_tag(m) for m in (selected_macros or [norm_primary]) if m]

    # 1. Check title first for unmistakable primary domain anchors
    matched_anchor = None
    for domain_cat, pattern in _PRIMARY_DOMAIN_ANCHORS:
        if re.search(pattern, title_low):
            matched_anchor = domain_cat
            break

    if matched_anchor:
        norm_primary = matched_anchor
        if matched_anchor not in macros:
            macros.insert(0, matched_anchor)
        else:
            macros.remove(matched_anchor)
            macros.insert(0, matched_anchor)

    # 2. Prune secondary macro tags that fail domain lexical guardrails
    cleaned_macros = [norm_primary]
    for m in macros:
        if m == norm_primary or m in cleaned_macros:
            continue
        guard_pat = _SECONDARY_TAG_GUARDS.get(m)
        if guard_pat and not re.search(guard_pat, lead_low):
            continue
        cleaned_macros.append(m)

    return norm_primary, cleaned_macros[:3]



TAG_LABEL_OVERRIDES = {
    "tsmc": "TSMC",
    "oled": "OLED",
    "vrr": "VRR",
    "variable_refresh_rate": "Variable Refresh Rate (VRR)",
    "jwst": "JWST (James Webb)",
    "ecb": "ECB (European Central Bank)",
    "2nm": "2nm Fabrication",
    "3nm": "3nm Fabrication",
    "gaafet": "GAAFET Transistors",
    "nanosheet": "Nanosheet Architecture",
    "apu": "Custom APU",
    "steamos": "SteamOS",
    "k2_18b": "K2-18b Exoplanet",
    "nist_standards": "NIST Standards",
    "nvidia": "NVIDIA",
    "steam_deck": "Steam Deck",
    "pc_gaming": "PC Gaming",
    "llm_agents": "LLM Agents",
    "ai_chips": "AI Chips & NPUs",
    "ai_ethics": "AI Safety & Ethics",
    "ai_tools": "Generative AI Tools",
    "ev_charging": "EV Charging & Range",
    "hardware_consoles": "Gaming Consoles (PS/Xbox/Switch)",
    "industry_esports": "Studios & Esports",
}

MACRO_METADATA: Dict[str, Dict[str, Any]] = {
    "Tech": {
        "display_name": "Tech & Software",
        "description": "General software, cloud computing, cybersecurity, web standards, IT, startups",
        "domain_keywords": ["post_quantum", "cryptography", "nist_standards", "runtime"],
    },
    "AI": {
        "display_name": "Artificial Intelligence",
        "description": "Large language models, autonomous agents, neural architectures, AI research & inference",
        "domain_keywords": ["autonomous_agents", "anthropic", "inference_optimization", "benchmarks", "nvidia"],
    },
    "Hardware": {
        "display_name": "Hardware & Semiconductors",
        "description": "Microchips, fabrication, TSMC, Intel, AMD, transistor processes, GPUs, memory",
        "domain_keywords": ["2nm", "3nm", "gaafet", "nanosheet", "tsmc", "transistors", "foundry", "silicon_wafers"],
    },
    "Automotive": {
        "display_name": "Automotive & Mobility",
        "description": "Electric vehicles, solid-state batteries, autonomous driving, robotaxis, automakers",
        "domain_keywords": ["solid_state_battery", "quantumscape", "porsche", "endurance_testing", "electric_vehicles"],
    },
    "Gaming": {
        "display_name": "Gaming & Handhelds",
        "description": "Video games, consoles, Steam Deck, PC gaming, graphics engines, game development",
        "domain_keywords": ["steam_deck", "oled", "handheld", "apu", "variable_refresh_rate", "steamos"],
    },
    "Finance": {
        "display_name": "Finance & Markets",
        "description": "Stock markets, cryptocurrency, macroeconomic policy, central banks, fintech, investing",
        "domain_keywords": ["digital_euro", "ecb", "settlement", "interbank"],
    },
    "Science": {
        "display_name": "Science & Space",
        "description": "Space exploration, astronomy, nuclear fusion, biotechnology, physics research",
        "domain_keywords": ["jwst", "exoplanet", "biosignatures", "k2_18b"],
    },
}


def format_tag_label(tag: str) -> str:
    """Formats raw tag into human-readable label."""
    if tag in TAG_LABEL_OVERRIDES:
        return TAG_LABEL_OVERRIDES[tag]
    return tag.replace("_", " ").title()


def get_taxonomy_hierarchy() -> List[Dict[str, Any]]:
    """Returns official hierarchy of Macro categories and their partitioned micro tags.
    
    Guarantees zero cross-domain tag contamination for the frontend onboarding popup.
    """
    hierarchy = []
    for macro in MACRO_CATEGORIES:
        subtags = TAXONOMY_CRITERIA.get(macro, {})
        meta = MACRO_METADATA.get(macro, {})
        keywords = meta.get("domain_keywords", [])

        # Combine criteria subtags and domain keywords without duplicates
        seen_tags = set()
        micro_tags_list = []

        # 1. Criteria subtags
        for tag in subtags.keys():
            if tag not in seen_tags and tag not in BANNED_MICRO_TAGS:
                seen_tags.add(tag)
                micro_tags_list.append({
                    "tag": tag,
                    "label": format_tag_label(tag)
                })

        # 2. Domain keywords belonging to this macro domain
        for tag in keywords:
            if tag not in seen_tags and tag not in BANNED_MICRO_TAGS:
                seen_tags.add(tag)
                micro_tags_list.append({
                    "tag": tag,
                    "label": format_tag_label(tag)
                })

        hierarchy.append({
            "macro": macro,
            "display_name": meta.get("display_name", macro),
            "description": meta.get("description", ""),
            "micro_tags": micro_tags_list
        })

    return hierarchy


