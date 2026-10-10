import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

ai_client = OpenAI(
    base_url=os.getenv("AI_BASE_URL"),
    api_key=os.getenv("AI_API_KEY")
)

FAST_MODEL = os.getenv("FAST_MODEL")
REASONING_MODEL = os.getenv("REASONING_MODEL")
