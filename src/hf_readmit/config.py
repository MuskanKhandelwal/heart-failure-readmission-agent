from __future__ import annotations

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into environment early to support local development.
load_dotenv(dotenv_path=Path(".env"))


class Settings(BaseSettings):
    """Application settings loaded from environment or .env file."""

    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    langfuse_public_key: Optional[str] = Field(default=None)
    langfuse_secret_key: Optional[str] = Field(default=None)
    langfuse_host: str = Field(default="https://cloud.langfuse.com")
    environment: str = Field(default="development")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
