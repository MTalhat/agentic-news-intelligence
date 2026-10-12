import requests
import sys
import json

sys.stdout.reconfigure(encoding='utf-8')
base = "http://localhost:8000"

print("================================================================")
print("             LIVE RSS-BACKEND ENDPOINT TEST REPORT              ")
print("================================================================\n")

# 1. Feed Endpoint
print("[1] GET /api/articles (Latest feed items):")
try:
    r = requests.get(f"{base}/api/articles", params={"page": 1, "page_size": 4})
    items = r.json()
    print(f"    Status: {r.status_code} OK | Returned: {len(items)} items")
    for a in items:
        print(f"    • [{a.get('macro_tag')}] {a.get('title')[:65]}")
        print(f"      Tags: {a.get('micro_tags')} | TLDR: {a.get('tldr', '')[:80]}...")
except Exception as e:
    print(f"    ERROR: {e}")

# 2. Macro Filtering
print("\n[2] GET /api/articles?macro=Hardware (Category filter):")
try:
    r = requests.get(f"{base}/api/articles", params={"macro": "Hardware"})
    items = r.json()
    print(f"    Status: {r.status_code} OK | Returned: {len(items)} hardware items")
    for a in items:
        print(f"    • {a.get('title')[:70]} | Tags: {a.get('micro_tags')}")
except Exception as e:
    print(f"    ERROR: {e}")

# 3. Search: "i want nanometer tech"
print("\n[3] GET /api/search?q=i want nanometer tech (Target query):")
try:
    r = requests.get(f"{base}/api/search", params={"q": "i want nanometer tech"})
    items = r.json()
    print(f"    Status: {r.status_code} OK | Returned: {len(items)} items")
    for a in items:
        print(f"    • {a.get('title')} | Tags: {a.get('micro_tags')}")
except Exception as e:
    print(f"    ERROR: {e}")

# 4. Search: "bitcoin"
print("\n[4] GET /api/search?q=bitcoin (Crypto search):")
try:
    r = requests.get(f"{base}/api/search", params={"q": "bitcoin"})
    items = r.json()
    print(f"    Status: {r.status_code} OK | Returned: {len(items)} items")
    for a in items:
        print(f"    • {a.get('title')} | Tags: {a.get('micro_tags')}")
except Exception as e:
    print(f"    ERROR: {e}")

# 5. Search: "solid state battery"
print("\n[5] GET /api/search?q=solid state battery (Automotive / Battery search):")
try:
    r = requests.get(f"{base}/api/search", params={"q": "solid state battery"})
    items = r.json()
    print(f"    Status: {r.status_code} OK | Returned: {len(items)} items")
    for a in items:
        print(f"    • {a.get('title')} | Tags: {a.get('micro_tags')}")
except Exception as e:
    print(f"    ERROR: {e}")

print("\n================================================================")
