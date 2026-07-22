import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
IMAGE_PROVIDER = os.getenv("IMAGE_PROVIDER", "")
IMAGE_API_KEY = os.getenv("IMAGE_API_KEY", "")
