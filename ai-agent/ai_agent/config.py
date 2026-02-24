from __future__ import annotations

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    anthropic_api_key: str = ""
    langsmith_api_key: str = ""
    langsmith_tracing: bool = True
    langsmith_project: str = "openemr-agent"

    openemr_base_url: str = "http://openemr:80"
    openemr_client_id: str = ""
    openemr_client_secret: str = ""
    openemr_username: str = "admin"
    openemr_password: str = "pass"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


def get_settings() -> Settings:
    """Return a Settings instance. Reads .env from ai-agent/ dir if present."""
    env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.isfile(env_file):
        return Settings(_env_file=env_file)
    return Settings()
