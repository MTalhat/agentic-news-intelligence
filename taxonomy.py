"""Centralized taxonomy and criteria definitions for hybrid AI triage and enrichment.

Synchronizes macro categories and micro-tags across ingestion, database GIN indexes,
and intent search.
"""

from typing import Dict, List, Optional

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

