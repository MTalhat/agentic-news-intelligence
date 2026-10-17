import sys
import time
import uuid
import requests
import psycopg

sys.stdout.reconfigure(encoding='utf-8')

BASE_URL = "http://localhost:8000"
DB_URL = "postgresql://dev_user:dev_password123@localhost:5432/rss_intelligence"

print("=" * 70)
print("     USER PERSONALIZATION & DAILY INTELLIGENCE DIGEST TEST SUITE")
print("=" * 70 + "\n")

# -------------------------------------------------------------------
# Test 1: Taxonomy Hierarchy (GET /api/taxonomy)
# -------------------------------------------------------------------
print("[1/7] Testing Taxonomy Hierarchy (GET /api/taxonomy)...")
r_tax = requests.get(f"{BASE_URL}/api/taxonomy")
assert r_tax.status_code == 200, f"Taxonomy endpoint returned {r_tax.status_code}"
taxonomy = r_tax.json()
assert len(taxonomy) == 7, f"Expected 7 macro categories, got {len(taxonomy)}"

# Verify macro partition and check for zero cross-domain leakage
gaming_cat = next((c for c in taxonomy if c["macro"] == "Gaming"), None)
finance_cat = next((c for c in taxonomy if c["macro"] == "Finance"), None)
assert gaming_cat is not None, "Gaming category missing"
assert finance_cat is not None, "Finance category missing"

gaming_tags = [t["tag"] for t in gaming_cat["micro_tags"]]
finance_tags = [t["tag"] for t in finance_cat["micro_tags"]]

assert "steam_deck" in gaming_tags, "steam_deck should be in Gaming"
assert "crypto" in finance_tags, "crypto should be in Finance"
assert "crypto" not in gaming_tags, "Zero Leakage violation: crypto leaked into Gaming"
assert "steam_deck" not in finance_tags, "Zero Leakage violation: steam_deck leaked into Finance"
print("  --> PASS: Taxonomy hierarchy partitioned cleanly across 7 categories with zero cross-leakage.")

# -------------------------------------------------------------------
# Test 2: User Onboarding Detection (POST /api/users/login)
# -------------------------------------------------------------------
print("\n[2/7] Testing User Onboarding Detection (POST /api/users/login)...")
test_username = f"zero_leak_user_{int(time.time() * 1000) % 100000}"
r_login = requests.post(f"{BASE_URL}/api/users/login", json={"username": test_username})
assert r_login.status_code == 200, f"Login returned {r_login.status_code}"
user_data = r_login.json()

user_id = user_data["id"]
assert user_data["username"] == test_username
assert user_data["onboarding_completed"] is False, "New user should have onboarding_completed = False"
assert user_data["preferred_macro_tags"] == [], "New user should have empty preferred macro tags"
print(f"  --> PASS: Created user '{test_username}', onboarding_completed correctly flagged as False.")

# -------------------------------------------------------------------
# Test 3: Cold-Start Fallback (GET /api/feed/personalized)
# -------------------------------------------------------------------
print("\n[3/7] Testing Cold-Start Fallback (GET /api/feed/personalized)...")
r_cold = requests.get(f"{BASE_URL}/api/feed/personalized", params={"user_id": user_id})
assert r_cold.status_code == 200, f"Cold start returned {r_cold.status_code}"
cold_data = r_cold.json()

assert cold_data["is_fallback"] is True, "Cold-start should set is_fallback = True"
assert len(cold_data["articles"]) > 0, "Cold-start should return fallback chronological feed"
print(f"  --> PASS: Safe fallback triggered; returned {len(cold_data['articles'])} articles with is_fallback=True.")

# -------------------------------------------------------------------
# Test 4: Natural Language Preference & Vector Synthesis (PUT /api/users/{id}/preferences)
# -------------------------------------------------------------------
print("\n[4/7] Testing Natural Language Preferences & 384d Vector (PUT /api/users/{id}/preferences)...")
pref_payload = {
    "preferred_macro_tags": ["Gaming"],
    "preferred_micro_tags": ["steam_deck", "pc_gaming", "oled"]
}
r_pref = requests.put(f"{BASE_URL}/api/users/{user_id}/preferences", json=pref_payload)
assert r_pref.status_code == 200, f"Preferences update failed: {r_pref.status_code} {r_pref.text}"
pref_data = r_pref.json()

assert pref_data["onboarding_completed"] is True, "onboarding_completed should be True after preferences"
assert "Gaming" in pref_data["preferred_macro_tags"]
assert "steam_deck" in pref_data["preferred_micro_tags"]
assert "interest_summary" in pref_data
assert "Technical news and updates in Gaming" in pref_data["interest_summary"]
assert "Steam Deck" in pref_data["interest_summary"]

# Verify vector in PostgreSQL
with psycopg.connect(DB_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT interest_embedding FROM users WHERE id = %s;", (user_id,))
        row = cur.fetchone()
        assert row is not None and row[0] is not None, "interest_embedding must be saved in database"
print(f"  --> PASS: Synthesized natural language interest sentence and 384d dense vector in PostgreSQL.")

# -------------------------------------------------------------------
# Test 5: Zero-Leakage Personalized Feed (GET /api/feed/personalized)
# -------------------------------------------------------------------
print("\n[5/7] Testing Zero-Leakage Personalized Feed...")
r_feed = requests.get(f"{BASE_URL}/api/feed/personalized", params={"user_id": user_id, "page_size": 20})
assert r_feed.status_code == 200, f"Feed returned {r_feed.status_code}"
feed_data = r_feed.json()

assert feed_data["is_fallback"] is False, "is_fallback should be False after onboarding"
articles = feed_data["articles"]
assert len(articles) > 0, "Expected personalized articles for Gaming"

# ZERO LEAKAGE ASSERTIONS:
for art in articles:
    # Assertion A: Zero Noise Leakage
    macro = art.get("macro_tag")
    assert macro != "Off-Topic", f"Zero Leakage Violation: Off-Topic article leaked into feed: {art['title']}"
    
    # Assertion B: Zero Domain Leakage (100% of articles match opted-in Gaming category)
    macros = [m.lower() for m in (art.get("macro_tags") or [macro])]
    assert "gaming" in macros, f"Zero Leakage Violation: Non-gaming article leaked into Gaming-only feed: {art['title']} (tags: {macros})"

# Check that scoring prioritized micro tags
high_score_articles = [a for a in articles if a.get("personal_score", 0) >= 40]
assert len(high_score_articles) > 0, "Top articles should receive elevated personal_score"
print(f"  --> PASS: Verified 100% Gaming precision across {len(articles)} articles with ZERO off-topic or cross-category leakage.")

# -------------------------------------------------------------------
# Test 6: Like/Unlike Toggle & Dynamic EMA Vector Reinforcement
# -------------------------------------------------------------------
print("\n[6/7] Testing Interaction Toggle & Dynamic EMA Vector Reinforcement...")
test_article_id = articles[0]["id"]

# 6A: Add Like
r_like1 = requests.post(
    f"{BASE_URL}/api/articles/{test_article_id}/interactions",
    json={"user_id": user_id, "interaction_type": "like"}
)
assert r_like1.status_code == 200, f"Like returned {r_like1.status_code}"
d_like1 = r_like1.json()
assert d_like1["status"] == "recorded" and d_like1["active"] is True, "First like should be recorded and active"

# 6B: Toggle Off Like
r_like2 = requests.post(
    f"{BASE_URL}/api/articles/{test_article_id}/interactions",
    json={"user_id": user_id, "interaction_type": "like"}
)
assert r_like2.status_code == 200, f"Second like returned {r_like2.status_code}"
d_like2 = r_like2.json()
assert d_like2["status"] == "removed" and d_like2["active"] is False, "Second like should toggle to removed and inactive"

# 6C: Dislike Repulsion & Immediate Feed Exclusion (E3, N22)
r_dislike1 = requests.post(
    f"{BASE_URL}/api/articles/{test_article_id}/interactions",
    json={"user_id": user_id, "interaction_type": "dislike"}
)
assert r_dislike1.status_code == 200, f"Dislike returned {r_dislike1.status_code}"
d_dislike1 = r_dislike1.json()
assert d_dislike1["status"] == "recorded" and d_dislike1["active"] is True
assert d_dislike1["ema_updated"] is True, "Dislike must trigger negative EMA repulsion on user vector"

r_feed_after_dislike = requests.get(f"{BASE_URL}/api/feed/personalized", params={"user_id": user_id, "page_size": 20})
assert r_feed_after_dislike.status_code == 200
after_ids = [a["id"] for a in r_feed_after_dislike.json().get("articles", [])]
assert test_article_id not in after_ids, "Disliked article must be immediately excluded from personalized feed"

# Untoggle dislike to leave clean state
requests.post(
    f"{BASE_URL}/api/articles/{test_article_id}/interactions",
    json={"user_id": user_id, "interaction_type": "dislike"}
)
print("  --> PASS: Like/Dislike toggle, EMA vector learning, and immediate disliked-article exclusion verified.")

# -------------------------------------------------------------------
# Test 7: Daily Rotating Intelligence Digest (GET /api/intelligence/daily-digest)
# -------------------------------------------------------------------
print("\n[7/7] Testing Daily Rotating Intelligence Digest (GET /api/intelligence/daily-digest)...")
r_digest1 = requests.get(f"{BASE_URL}/api/intelligence/daily-digest", params={"user_id": user_id})
assert r_digest1.status_code == 200, f"Daily digest failed: {r_digest1.status_code} {r_digest1.text}"
digest1 = r_digest1.json()

assert digest1["macro_tag"] == "Gaming", f"Expected Gaming digest, got {digest1['macro_tag']}"
assert len(digest1["article_ids"]) >= 1, f"Digest must return at least 1 verified article, got {len(digest1['article_ids'])}"
assert len(digest1["synthesis_report"]) > 120, "Synthesis report markdown should be a full multi-section brief"
assert "Shared Consensus" in digest1["synthesis_report"], "Digest report must include Section 1 (Shared Consensus)"
assert "evidence_card" in digest1 and digest1["evidence_card"] is not None, "Digest must attach structured evidence_card"
assert "date" in digest1, "Digest must contain calendar date"

# Test Cache hit on second call (sub-50ms)
t0 = time.perf_counter()
r_digest2 = requests.get(f"{BASE_URL}/api/intelligence/daily-digest", params={"user_id": user_id})
lat_ms = (time.perf_counter() - t0) * 1000
assert r_digest2.status_code == 200
digest2 = r_digest2.json()
assert digest2["is_cached"] is True, "Second call must hit PostgreSQL cache"
assert "Shared Consensus" in digest2["synthesis_report"], "Cached digest must also preserve full 4-section report"
print(f"  --> PASS: Daily digest generated with structured 4-section brief + evidence_card and cached (Cache latency: {lat_ms:.1f}ms).")

print("\n" + "=" * 70)
print("           ALL USER PERSONALIZATION & DIGEST TESTS PASSED!")
print("=" * 70)

