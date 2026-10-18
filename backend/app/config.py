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
    # Which M3 weights file in app/ai/weights/ to load: the real PetFinder model, or "m3_synthetic"
    # (the simulated-data experiment from ml/m3_synthetic) for comparison.
    m3_weights: str = "m3_adoption_speed"

    # Age model (ONNX, with its .json config next to it). Relative paths start at the project root.
    m4_age_model_path: str = "backend/models/m4_age.onnx"
    # Dog skin model (M5), same rules as above.
    m5_skin_model_path: str = "backend/models/m5_skin.onnx"

    # M2 photo-matching service (ml/photo_matching), used by app.ai.models:PawModels.
    m2_url: str = "http://127.0.0.1:8002"
    m2_timeout: float = 20.0

    # M7 (lost-pet location): folder with the trained model and map grids, relative to backend/.
    m7_dir: str = "../ml/m7_location"

    # Reference point used for "distance from you" when the visitor has no location.
    home_lat: float = 36.8782
    home_lng: float = 10.3247


@lru_cache
def get_settings() -> Settings:
    return Settings()
