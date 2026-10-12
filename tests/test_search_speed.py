import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import natural_search

sys.stdout.reconfigure(encoding="utf-8")

queries = [
    "i want semiconductor news",
    "i want nanometer tech",
    "bitcoin",
    "solid state battery",
    "steam deck",
    "autonomous agent"
]

print("=" * 70)
print("       STREAMLINED INSTANT SEARCH BENCHMARK & ACCURACY REPORT")
print("=" * 70)

for q in queries:
    t0 = time.perf_counter()
    results = natural_search(q)
    latency = (time.perf_counter() - t0) * 1000
    print(f"\nQuery: '{q}' (Latency: {latency:.2f} ms | Found: {len(results)})")
    for r in results[:3]:
        print(f"  • {r['title']}")
        print(f"    Tags: {r['micro_tags']}")

print("\n" + "=" * 70)
