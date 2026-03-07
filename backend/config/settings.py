import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY")

settings = Settings()