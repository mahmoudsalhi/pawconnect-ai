from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/pawconnect"
    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60 * 24 * 7

    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    max_upload_mb: int = 10

    # Which AI implementation to load. "demo" is the built-in placeholder; point this at
    # your own provider class (e.g. "app.ai.custom:MyProvider") once the models are ready.
    ai_provider: str = "demo"
    # M7 (lost-pet location): folder with the trained model and map grids, relative to backend/.
    m7_dir: str = "../ml/m7_location"
    # AI search assistant. "auto" uses the first key found (Groq, then Anthropic); no key = M7 only.
    ai_assistant: str = "auto"
    groq_api_key: str = ""                       # free tier, no credit card: console.groq.com
    groq_model: str = "openai/gpt-oss-120b"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-5-5"

    # Reference point used for "distance from you" when the visitor has no location.
    home_lat: float = 36.8782
    home_lng: float = 10.3247


@lru_cache
def get_settings() -> Settings:
    return Settings()
