from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import model_validator


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    MONGODB_URI: str = ""
    DB_NAME: str = "rag_db"
    COLLECTION_NAME: str = "documents"
    COSTS_COLLECTION: str = "query_costs"

    GEMINI_API_KEY: str = ""
    SIMPLE_MODEL: str = "gemini-1.5-flash"
    COMPLEX_MODEL: str = "gemini-1.5-pro"
    COMPLEXITY_THRESHOLD: float = 0.4

    REDIS_URL: str = "redis://redis:6379"
    CACHE_TTL: int = 3600
    CACHE_SIMILARITY_THRESHOLD: float = 0.92

    API_KEY: str = ""
    LOG_LEVEL: str = "INFO"

    @model_validator(mode="after")
    def _require_secrets(self) -> "Settings":
        missing = [k for k, v in {"GEMINI_API_KEY": self.GEMINI_API_KEY, "MONGODB_URI": self.MONGODB_URI}.items() if not v]
        if missing:
            raise ValueError(f"missing required env vars: {', '.join(missing)}")
        return self


cfg = Settings()
