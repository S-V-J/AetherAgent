"""
AetherAgent Application Settings.
All configuration loaded from environment variables or .env file.
"""

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # --- App ---
    app_name: str = "AetherAgent"
    app_version: str = "0.1.0"
    debug: bool = False
    api_prefix: str = "/api/v1"

    # --- Database (SQLite for dev, PostgreSQL for production) ---
    database_url: str = "sqlite+aiosqlite:///data/aetheragent.db"

    # --- JWT Authentication ---
    jwt_secret_key: str = "aetheragent-dev-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 60
    jwt_refresh_token_expire_days: int = 7

    # --- Paths ---
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = Path("data")
    models_dir: Path = Path("data/models")
    uploads_dir: Path = Path("data/uploads")
    vector_db_dir: Path = Path("data/vector_db")
    runtime_config_path: Path = Path("runtime_config.json")

    # --- Inference ---
    model_path: Optional[str] = None  # Path to GGUF model file
    default_system_prompt: str = (
        "You are AetherAgent, a powerful local AI assistant. "
        "Respond in Markdown format. Be concise, accurate, and helpful."
    )

    # --- Security Module ---
    security_module_enabled: bool = False

    # --- CORS ---
    cors_origins: list[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resolve relative paths against base_dir
        if not self.data_dir.is_absolute():
            self.data_dir = self.base_dir / self.data_dir
        if not self.models_dir.is_absolute():
            self.models_dir = self.base_dir / self.models_dir
        if not self.uploads_dir.is_absolute():
            self.uploads_dir = self.base_dir / self.uploads_dir
        if not self.vector_db_dir.is_absolute():
            self.vector_db_dir = self.base_dir / self.vector_db_dir
        if not self.runtime_config_path.is_absolute():
            self.runtime_config_path = self.base_dir / self.runtime_config_path

    def ensure_directories(self):
        """Create all required data directories."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db_dir.mkdir(parents=True, exist_ok=True)


# Singleton instance
settings = Settings()
