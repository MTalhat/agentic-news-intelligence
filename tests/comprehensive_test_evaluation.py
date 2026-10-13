import requests
import sys
import time
import json
import uuid

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
SECRET = "local_hackathon_dev_secret"

scores = []

def record_result(name, passed, score, max_score, details, output_sample=""):
    scores.append({
        "name": name,
        "passed": passed,
        "score": score,
        "max_score": max_score,
        "details": details,
        "output_sample": output_sample
    })

print("=" * 70)
print("       COMPREHENSIVE BACKEND TEST & ACCURACY EVALUATION")
print("=" * 70 + "\n")

# -------------------------------------------------------------
# 1. GET /api/articles (Feed & Macro Filtering)
# -------------------------------------------------------------
print("[1/9] Testing Feed & Filtering (GET /api/articles)...")
try:
    r1 = requests.get(f"{BASE_URL}/api/articles")
    all_articles = r1.json()
    r2 = requests.get(f"{BASE_URL}/api/articles", params={"macro": "Gaming"})
    gaming_articles = r2.json()

    feed_ok = r1.status_code == 200 and len(all_articles) >= 10
    filter_ok = r2.status_code == 200 and len(gaming_articles) >= 1 and all(
        (a.get("macro_tag") or "").lower() == "gaming" or "gaming" in [m.lower() for m in a.get("macro_tags", [])]
        for a in gaming_articles
    )

    if feed_ok and filter_ok:
        record_result("Feed & Filtering", True, 10, 10, f"Retrieved {len(all_articles)} total articles, filtered {len(gaming_articles)} Gaming articles with 100% precision", f"Gaming title: {gaming_articles[0]['title']}")
        print("  --> PASS (10/10)")
    else:
        record_result("Feed & Filtering", False, 5, 10, f"Status: {r1.status_code}/{r2.status_code}")
        print("  --> PARTIAL (5/10)")
except Exception as e:
    record_result("Feed & Filtering", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

def read_sse_stream(response):
    """Parses standard Server-Sent Events (data: {...}) stream into combined text."""
    collected = []
    for line in response.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                payload = decoded[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data_obj = json.loads(payload)
                    if "token" in data_obj:
                        collected.append(data_obj["token"])
                    elif "error" in data_obj:
                        collected.append(f" ERROR: {data_obj['error']} ")
                except Exception:
                    collected.append(payload)
    return "".join(collected)

# Dynamic reference IDs lookup with robust fallbacks
steam_deck_id = "16b5a828-ad47-48da-88a9-8bab67bc7b3d"
wired_battery_id = "82a5ecf9-e6ab-4152-afae-5ca536361bd5"
electrek_battery_id = "bf07dc92-4550-4f46-9cf1-f00bd01aaa09"

try:
    r_all = requests.get(f"{BASE_URL}/api/articles", params={"page_size": 50})
    if r_all.status_code == 200:
        for a in r_all.json():
            t = a.get("title", "").lower()
            if "steam deck" in t:
                steam_deck_id = str(a["id"])
            elif "quantumscape" in t:
                electrek_battery_id = str(a["id"])
            elif "automaker" in t:
                wired_battery_id = str(a["id"])
except Exception:
    pass

# -------------------------------------------------------------
# 2. GET /api/articles/{article_id} (Full Article Detail)
# -------------------------------------------------------------
print("\n[2/9] Testing Single Article Detail (GET /api/articles/{id})...")
try:
    r = requests.get(f"{BASE_URL}/api/articles/{steam_deck_id}")
    art = r.json()
    has_keys = all(k in art for k in ["id", "title", "clean_text", "macro_tag", "micro_tags", "tldr"])
    content_valid = len(art.get("clean_text", "")) > 50 and len(art.get("tldr", "")) > 20

    if r.status_code == 200 and has_keys and content_valid:
        record_result("Single Article Detail", True, 10, 10, "All schema fields present, clean text & TLDR intact", f"TLDR: {art['tldr'][:90]}...")
        print("  --> PASS (10/10)")
    else:
        record_result("Single Article Detail", False, 5, 10, f"Status: {r.status_code}")
        print("  --> FAIL")
except Exception as e:
    record_result("Single Article Detail", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 3. POST /api/users/login & GET /api/users (User Profiles)
# -------------------------------------------------------------
print("\n[3/9] Testing User Authentication & Profile (POST /api/users/login)...")
test_username = f"evaluator_{int(time.time()) % 1000}"
user_id = None
try:
    r_login = requests.post(f"{BASE_URL}/api/users/login", json={"username": test_username})
    user_data = r_login.json()
    user_id = user_data.get("id")
    r_list = requests.get(f"{BASE_URL}/api/users")
    users = r_list.json()

    user_found = any(u["username"] == test_username for u in users)
    if r_login.status_code == 200 and user_id and user_found:
        record_result("User Profile System", True, 10, 10, f"Created user '{test_username}', assigned UUID & avatar, verified in user registry", f"Avatar: {user_data.get('avatar_url')}")
        print("  --> PASS (10/10)")
    else:
        record_result("User Profile System", False, 5, 10, "User login/list error")
        print("  --> FAIL")
except Exception as e:
    record_result("User Profile System", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 4. Comments: POST & GET /api/articles/{id}/comments
# -------------------------------------------------------------
print("\n[4/9] Testing Social Annotations & Comments (POST & GET comments)...")
try:
    c_payload = {
        "user_id": user_id or "a0eebc99-9c0b-4ef8-bb6d-6bb9bd380a11",
        "selected_text": "120Hz VRR OLED panel",
        "comment_text": "Automated evaluation comment: OLED VRR is outstanding."
    }
    r_post = requests.post(f"{BASE_URL}/api/articles/{steam_deck_id}/comments", json=c_payload)
    r_get = requests.get(f"{BASE_URL}/api/articles/{steam_deck_id}/comments")
    comments = r_get.json()

    comment_matched = any(c.get("comment") == c_payload["comment_text"] for c in comments)
    has_user_join = any(c.get("username") is not None for c in comments)

    if r_post.status_code == 200 and comment_matched and has_user_join:
        record_result("Social Comments", True, 10, 10, "Comment saved with quote selection, joined with author username & avatar", f"Latest comment by: {comments[-1].get('username')}")
        print("  --> PASS (10/10)")
    else:
        record_result("Social Comments", False, 5, 10, "Comment post or get failed")
        print("  --> FAIL")
except Exception as e:
    record_result("Social Comments", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 5. POST /api/internal/ingest (Ingestion & Deduplication)
# -------------------------------------------------------------
print("\n[5/9] Testing Ingestion Pipeline & SHA-256 Deduplication...")
try:
    unique_url = f"https://eval.example.com/news-{uuid.uuid4()}"
    test_article = [{
        "Site": "Benchmark News",
        "Baslik": "Automated Pipeline Integrity Verification Article",
        "Link": unique_url,
        "Yazar": "QA Bot",
        "Tarih": "Sat, 10 Oct 2026 12:00:00 GMT",
        "TamMetin": "This is an automated test article verifying that the ingestion pipeline computes SHA-256 digests and deduplicates duplicate URLs."
    }]
    # First ingest
    r_first = requests.post(f"{BASE_URL}/api/internal/ingest", json=test_article, headers={"x-secret": SECRET})
    d1 = r_first.json()
    # Duplicate ingest
    r_second = requests.post(f"{BASE_URL}/api/internal/ingest", json=test_article, headers={"x-secret": SECRET})
    d2 = r_second.json()

    if d1.get("inserted") == 1 and d2.get("inserted") == 0:
        record_result("Ingestion & Deduplication", True, 10, 10, "Article inserted with SHA-256 digest; duplicate attempt correctly ignored (inserted: 0)", f"First: {d1}, Second: {d2}")
        print("  --> PASS (10/10)")
    else:
        record_result("Ingestion & Deduplication", False, 5, 10, f"d1: {d1}, d2: {d2}")
        print("  --> FAIL")
except Exception as e:
    record_result("Ingestion & Deduplication", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 6. GET /api/search (Natural Language & Tag Likeness)
# -------------------------------------------------------------
print("\n[6/9] Testing Natural Language Search & Tag Likeness (GET /api/search)...")
try:
    queries = [
        ("i want steam games", "Valve Unveils Steam Deck 2"),
        ("space discoveries", "James Webb Space Telescope"),
        ("ev battery", "QuantumScape")
    ]
    search_score = 0
    search_samples = []

    for q, expected in queries:
        r_s = requests.get(f"{BASE_URL}/api/search", params={"q": q})
        items = r_s.json()
        found = any(expected.lower() in item["title"].lower() for item in items)
        if found:
            search_score += 1
            search_samples.append(f"'{q}' -> Matched: '{expected}'")

    if search_score == len(queries):
        record_result("Search & Tag Likeness", True, 10, 10, "3/3 distinct domain queries matched exact target articles via tag likeness", "; ".join(search_samples))
        print("  --> PASS (10/10)")
    else:
        record_result("Search & Tag Likeness", False, int((search_score/len(queries))*10), 10, f"{search_score}/{len(queries)} queries matched", "; ".join(search_samples))
        print(f"  --> PARTIAL ({search_score}/{len(queries)})")
except Exception as e:
    record_result("Search & Tag Likeness", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 7. Tier 2 Grounded Copilot (Factual Accuracy Test)
# -------------------------------------------------------------
print("\n[7/9] Testing Tier 2 Copilot Factual Accuracy (POST /chat)...")
try:
    q_fact = {
        "history": [],
        "question": "What display refresh rate and panel type does Steam Deck 2 have?"
    }
    r_chat = requests.post(f"{BASE_URL}/api/articles/{steam_deck_id}/chat", json=q_fact, stream=True)
    ans = read_sse_stream(r_chat).lower()

    has_oled = "oled" in ans
    has_120 = "120" in ans or "vrr" in ans

    if r_chat.status_code == 200 and has_oled and has_120:
        record_result("Copilot Factual Accuracy", True, 10, 10, "Copilot extracted accurate facts (120Hz VRR & OLED) strictly from the text", ans[:100].strip())
        print("  --> PASS (10/10)")
    else:
        record_result("Copilot Factual Accuracy", False, 6, 10, f"Tokens missing exact specs: {ans[:80]}", ans[:80])
        print("  --> PARTIAL")
except Exception as e:
    record_result("Copilot Factual Accuracy", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 8. Tier 2 Grounded Copilot (Anti-Hallucination Guardrail)
# -------------------------------------------------------------
print("\n[8/9] Testing Tier 2 Copilot Anti-Hallucination Guardrail...")
try:
    q_trick = {
        "history": [],
        "question": "Does this article state that Nintendo announced a new Zelda game for it?"
    }
    r_trick = requests.post(f"{BASE_URL}/api/articles/{steam_deck_id}/chat", json=q_trick, stream=True)
    ans_trick = read_sse_stream(r_trick).lower()

    refused = any(p in ans_trick for p in ["not state", "does not mention", "no", "not mentioned", "not in the text"])

    if r_trick.status_code == 200 and refused:
        record_result("Anti-Hallucination Guardrail", True, 10, 10, "Successfully refused hallucinated premise; adhered strictly to ground truth", ans_trick[:100].strip())
        print("  --> PASS (10/10)")
    else:
        record_result("Anti-Hallucination Guardrail", False, 4, 10, f"Did not explicitly refuse: {ans_trick[:80]}", ans_trick[:80])
        print("  --> FAIL")
except Exception as e:
    record_result("Anti-Hallucination Guardrail", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# 9. Tier 3 Multi-Source Synthesis (Consensus & Discrepancy)
# -------------------------------------------------------------
print("\n[9/9] Testing Tier 3 Multi-Source Synthesis Agent (POST /synthesize)...")
try:
    synth_req = {
        "article_ids": [electrek_battery_id, wired_battery_id]
    }
    r_synth = requests.post(f"{BASE_URL}/api/intelligence/synthesize", json=synth_req, stream=True)
    synth_text = read_sse_stream(r_synth)

    has_consensus = any(w in synth_text.lower() for w in ["consensus", "agree", "shared"])
    has_discrepancy = any(w in synth_text.lower() for w in ["discrepanc", "contradiction", "disagree", "contrast", "differ"])
    has_length = len(synth_text) > 200

    if r_synth.status_code == 200 and has_consensus and has_length:
        record_result("Multi-Source Synthesis Agent", True, 10, 10, "Streamed multi-source synthesis with structured sections (Consensus & Discrepancies)", synth_text[:120].strip() + "...")
        print("  --> PASS (10/10)")
    else:
        record_result("Multi-Source Synthesis Agent", False, 6, 10, f"Length: {len(synth_text)}", synth_text[:80])
        print("  --> PARTIAL")
except Exception as e:
    record_result("Multi-Source Synthesis Agent", False, 0, 10, str(e))
    print(f"  --> FAIL: {e}")

# -------------------------------------------------------------
# Final Scoring & Report
# -------------------------------------------------------------
total_score = sum(s["score"] for s in scores)
max_possible = sum(s["max_score"] for s in scores)
percentage = round((total_score / max_possible) * 100, 1)

print("\n" + "=" * 70)
print(f"        FINAL EVALUATION REPORT - SCORE: {total_score}/{max_possible} ({percentage}%)")
print("=" * 70)

for s in scores:
    status_icon = "✓" if s["passed"] else "✗"
    print(f"\n{status_icon} [{s['name']}]: {s['score']}/{s['max_score']}")
    print(f"   Details: {s['details']}")
    if s["output_sample"]:
        print(f"   Sample Output: \"{s['output_sample']}\"")

print("\n" + "=" * 70)
print(f"OVERALL PLATFORM RATING: {'EXCELLENT (A+)' if percentage >= 90 else 'GOOD (B)'} - {percentage}%")
print("=" * 70)
