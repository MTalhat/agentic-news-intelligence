import requests
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

test_queries = [
    "gaming",
    "game",
    "i want steam games",
    "ev battery",
    "space discoveries",
    "digital euro",
    "quantum encryption"
]

print("=" * 65)
print("              LIVE SEARCH TEST SUITE EXECUTION")
print("=" * 65 + "\n")

for idx, q in enumerate(test_queries, 1):
    start = time.time()
    resp = requests.get("http://localhost:8000/api/search", params={"q": q})
    elapsed = round(time.time() - start, 2)
    
    print(f"[{idx}/{len(test_queries)}] Query: \"{q}\"  (Status: {resp.status_code}, Time: {elapsed}s)")
    if resp.status_code == 200:
        items = resp.json()
        print(f"Total Matches: {len(items)}")
        for item in items:
            print(f"  • [{item['source']}] {item['title']}")
            print(f"    Tags: {item['micro_tags']}")
            print(f"    TLDR: {item['tldr'][:120]}...")
    else:
        print(f"  Error: {resp.text}")
    print("-" * 65)
