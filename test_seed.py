import requests

API_URL = "http://localhost:8000/api/internal/ingest"
SECRET = "local_hackathon_dev_secret"

mock_payload = [
    {
        "Site": "TechCrunch",
        "Baslik": "NVIDIA Unveils New Open Inference Optimization Runtimes",
        "Link": "https://techcrunch.com/2026/10/08/nvidia-inference-runtimes",
        "Yazar": "Alex Heath",
        "Tarih": "Thu, 08 Oct 2026 12:00:00 GMT",
        "TamMetin": "NVIDIA announced new high-throughput inference runtimes designed to optimize low-latency token generation on open hardware clusters. The software focuses on KV-cache compression and multi-GPU memory throughput, drastically lowering local compute requirements."
    },
    {
        "Site": "The Verge",
        "Baslik": "NVIDIA Claims 3x Speedup for Open Models, Benchmarks Questioned",
        "Link": "https://theverge.com/2026/10/08/nvidia-open-models-claims",
        "Yazar": "Nilay Patel",
        "Tarih": "Thu, 08 Oct 2026 13:15:00 GMT",
        "TamMetin": "NVIDIA stated today that its new model runtimes provide up to 3x inference speedups on enterprise cards. However, early independent testers noted that these gains only apply under specific batch sizes, with memory latency remaining standard under consumer workloads."
    },
    {
        "Site": "Wired",
        "Baslik": "Automakers Accelerate Next-Gen Solid-State Battery Roadmaps",
        "Link": "https://wired.com/2026/10/08/ev-solid-state-batteries",
        "Yazar": "Steven Levy",
        "Tarih": "Thu, 08 Oct 2026 10:00:00 GMT",
        "TamMetin": "Global electric vehicle manufacturers are moving up deployment timelines for solid-state batteries. New silicon-anode chemistries promise 600-mile ranges with charging times cut to ten minutes under proper grid infrastructure."
    },
    {
        "Site": "Ars Technica",
        "Baslik": "Anthropic Releases Advanced Autonomous Agent Framework for Developer Workstations",
        "Link": "https://arstechnica.com/ai/2026/10/anthropic-autonomous-agent-framework",
        "Yazar": "Benj Edwards",
        "Tarih": "Fri, 09 Oct 2026 14:30:00 GMT",
        "TamMetin": "Anthropic today introduced a workstation-native autonomous agent runtime that allows developers to run complex, multi-step coding and system management workflows locally. The framework introduces deterministic tool verification, sandboxed memory spaces, and built-in guardrails against unauthorized file system mutations."
    },
    {
        "Site": "Polygon",
        "Baslik": "Valve Unveils Steam Deck 2 With Custom 3nm APU and Variable Refresh OLED",
        "Link": "https://polygon.com/gaming/2026/10/valve-steam-deck-2-announcement",
        "Yazar": "Christopher Plante",
        "Tarih": "Fri, 09 Oct 2026 16:00:00 GMT",
        "TamMetin": "Valve officially revealed the Steam Deck 2, boasting a custom AMD monolithic chip fabricated on TSMC's 3nm process. The handheld gaming console features a 120Hz VRR OLED panel, 60 percent longer battery life, and complete backward compatibility with SteamOS libraries without requiring translation layer modifications."
    },
    {
        "Site": "Bloomberg",
        "Baslik": "European Central Bank Completes First Commercial Settlement on Wholesale Digital Euro Network",
        "Link": "https://bloomberg.com/news/articles/2026-10-09/ecb-wholesale-digital-euro-pilot-success",
        "Yazar": "Jana Randow",
        "Tarih": "Fri, 09 Oct 2026 09:45:00 GMT",
        "TamMetin": "The European Central Bank announced the successful settlement of a 500 million euro interbank bond transaction using its distributed ledger wholesale digital currency prototype. European finance ministers commended the pilot as a major milestone toward sovereign digital liquidity and reduced cross-border friction."
    },
    {
        "Site": "Nature",
        "Baslik": "James Webb Space Telescope Identifies Atmospheric Biosignature Markers on Exoplanet K2-18b",
        "Link": "https://nature.com/articles/s41586-026-08122-space-biosignature",
        "Yazar": "Dr. Elena Rostova",
        "Tarih": "Thu, 08 Oct 2026 18:20:00 GMT",
        "TamMetin": "Astrophysicists analyzing spectroscopic data from the James Webb Space Telescope have confirmed statistically significant absorption features of dimethyl sulfide and methane in the atmosphere of sub-Neptune exoplanet K2-18b. Researchers emphasize that while biological origins remain unproven, the findings provide unprecedented evidence of a water-rich, temperate world."
    },
    {
        "Site": "Tom's Hardware",
        "Baslik": "TSMC Achieves Over 80 Percent Yield on Next-Generation 2nm GAAFET Nodes",
        "Link": "https://tomshardware.com/news/tsmc-2nm-gaafet-production-yield-milestone",
        "Yazar": "Paul Alcorn",
        "Tarih": "Fri, 09 Oct 2026 11:10:00 GMT",
        "TamMetin": "TSMC announced during its quarterly technology symposium that its N2 nanosheet architecture has achieved defect densities low enough for over 80 percent production yields. Major hardware partners including Apple, AMD, and NVIDIA have reportedly committed volume orders for flagship processors scheduled for late 2027 delivery."
    },
    {
        "Site": "Electrek",
        "Baslik": "QuantumScape and Porsche Complete 500,000-Mile Real-World Solid-State Endurance Test",
        "Link": "https://electrek.co/2026/10/09/quantumscape-porsche-solid-state-endurance",
        "Yazar": "Fred Lambert",
        "Tarih": "Fri, 09 Oct 2026 13:00:00 GMT",
        "TamMetin": "Following two years of track and fleet evaluation, Porsche confirmed that QuantumScape's ceramic-separator solid-state battery cells retained more than 95 percent of original capacity after half a million simulated test miles. The testing showed zero dendrite formation and stable fast-charging performance across extreme ambient temperature regimes."
    },
    {
        "Site": "Reuters",
        "Baslik": "NIST Finalizes Global Post-Quantum Cryptography Standards for Financial Infrastructure",
        "Link": "https://reuters.com/technology/cybersecurity/nist-post-quantum-cryptography-standards-final-2026",
        "Yazar": "Raphael Satter",
        "Tarih": "Thu, 08 Oct 2026 15:40:00 GMT",
        "TamMetin": "The US National Institute of Standards and Technology has published its finalized specifications for quantum-resistant encryption algorithms, mandating implementation across federal agencies and global banking rails. The new lattice-based standards are engineered to protect critical public key infrastructure against theoretical decrypt-now attacks by future quantum computers."
    }
]

response = requests.post(
    API_URL,
    json=mock_payload,
    headers={"x-secret": SECRET}
)
print("Seeder Response:", response.status_code, response.json())
