import requests
import sys

sys.stdout.reconfigure(encoding='utf-8')

test_cases = [
    {
        "title": "Use Case 1: Direct Fact Extraction",
        "article_id": "82a5ecf9-e6ab-4152-afae-5ca536361bd5",
        "body": {
            "history": [],
            "question": "What are the claimed charging times and ranges?"
        }
    },
    {
        "title": "Use Case 2: Hallucination & Fact-Checking Guardrail",
        "article_id": "2b99bbe8-22ec-4938-97c0-2cb2d025fd66",
        "body": {
            "history": [],
            "question": "Does this article mention Tesla or Apple?"
        }
    },
    {
        "title": "Use Case 3: Technical Details Extraction",
        "article_id": "2b99bbe8-22ec-4938-97c0-2cb2d025fd66",
        "body": {
            "history": [],
            "question": "What specific techniques are used to lower compute requirements?"
        }
    },
    {
        "title": "Use Case 4: Critical Reading & Caveats",
        "article_id": "eba0b28d-baab-49ff-a32e-687553cebc1d",
        "body": {
            "history": [],
            "question": "What doubts or criticisms did independent testers raise?"
        }
    },
    {
        "title": "Use Case 5: Multi-Turn Conversation (with History)",
        "article_id": "eba0b28d-baab-49ff-a32e-687553cebc1d",
        "body": {
            "history": [
                {"role": "user", "content": "What is the claimed speedup?"},
                {"role": "assistant", "content": "NVIDIA claims up to a 3x inference speedup for enterprise models."}
            ],
            "question": "Under what conditions does that 3x speedup actually apply?"
        }
    }
]

results = []

import json

# Dynamically find relevant rich articles from database feed
try:
    r_feed = requests.get("http://localhost:8000/api/articles", params={"page_size": 50})
    if r_feed.status_code == 200 and r_feed.json():
        articles = r_feed.json()
        def find_id(kw, fallback_id):
            for a in articles:
                if kw.lower() in a.get("title", "").lower():
                    return str(a["id"])
            return fallback_id

        porsche_id = find_id("quantumscape", "bf07dc92-4550-4f46-9cf1-f00bd01aaa09")
        nvidia_id = find_id("nvidia", "eba0b28d-baab-49ff-a32e-687553cebc1d")
        deck_id = find_id("steam deck", "16b5a828-ad47-48da-88a9-8bab67bc7b3d")

        test_cases[0]["article_id"] = porsche_id
        test_cases[1]["article_id"] = deck_id
        test_cases[2]["article_id"] = nvidia_id
        test_cases[3]["article_id"] = nvidia_id
        test_cases[4]["article_id"] = nvidia_id
except Exception:
    pass

for idx, tc in enumerate(test_cases, 1):
    header = f"[{idx}/5] {tc['title']}\nArticle ID: {tc['article_id']}\nQuestion: {tc['body']['question']}"
    if tc['body']['history']:
        header += f"\nPrior Turns in History: {len(tc['body']['history'])}"
    print(header, flush=True)

    url = f"http://localhost:8000/api/articles/{tc['article_id']}/chat"
    resp = requests.post(url, json=tc['body'], stream=True)
    
    output_tokens = []
    for line in resp.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            if decoded.startswith("data: "):
                payload = decoded[6:].strip()
                if payload == "[DONE]":
                    break
                try:
                    data_obj = json.loads(payload)
                    if "token" in data_obj:
                        output_tokens.append(data_obj["token"])
                    elif "error" in data_obj:
                        output_tokens.append(f" [ERROR: {data_obj['error']}] ")
                except Exception:
                    output_tokens.append(payload)
    output = "".join(output_tokens).strip()
    
    print(f"Copilot Output:\n{output}\n", flush=True)
    results.append(f"{header}\n\nCopilot Output:\n{output}\n" + "-"*50)

with open("copilot_test_results.txt", "w", encoding="utf-8") as f:
    f.write("\n\n".join(results))

print("ALL_TESTS_COMPLETED_SUCCESSFULLY", flush=True)
