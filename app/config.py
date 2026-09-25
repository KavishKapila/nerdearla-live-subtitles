"""
Central configuration for the Nerdearla Vibeathon live subtitles backend.

Reads everything from environment variables (with sensible defaults) so the
service can be reconfigured at deploy time. Copy .env.example to .env and
adjust as needed -- python-dotenv loads it automatically on import.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _list(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# Project root = two levels up from app/config.py (where run.py lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _default_firebase_path() -> str:
    """Firebase service-account JSON, resolved relative to the project root
    so it's found regardless of where the server is launched from."""
    return str(_PROJECT_ROOT / "firebase-service-account.json")


@dataclass
class Settings:
    # ---- Speech-to-text --------------------------------------------------
    TRANSCRIPTION_ENGINE: str = field(
        default_factory=lambda: os.getenv("TRANSCRIPTION_ENGINE", "faster_whisper")
    )
    WHISPER_MODEL_SIZE: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small")
    )
    WHISPER_DEVICE: str = field(
        default_factory=lambda: os.getenv("WHISPER_DEVICE", "auto")
    )
    WHISPER_COMPUTE_TYPE: str = field(
        default_factory=lambda: os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    )
    WHISPER_BEAM_SIZE: int = field(
        default_factory=lambda: int(os.getenv("WHISPER_BEAM_SIZE", "1"))
    )
    WHISPER_INITIAL_PROMPT: str = field(default_factory=lambda: os.getenv(
        "WHISPER_INITIAL_PROMPT",
        "Nerdearla developer conference. Terms: API, Python, FastAPI, WebSocket, "
        "async, asyncio, uvicorn, Docker, Kubernetes, React, TypeScript, LLM, "
        "transformer, Whisper, embedding, vector database, CTranslate2, CUDA, GPU, "
        "Nerdearla, Vibeathon, hackathon, prompt, agent, token, latency, streaming."
    ))

    # ---- Translation (DeepL only) ----------------------------------------
    TRANSLATION_ENGINE: str = field(
        default_factory=lambda: os.getenv("TRANSLATION_ENGINE", "deepl")
    )
    DEEPL_API_KEY: str | None = field(
        default_factory=lambda: os.getenv("DEEPL_API_KEY")
    )
    TRANSLATION_BATCH_SIZE: int = field(
        default_factory=lambda: int(os.getenv("TRANSLATION_BATCH_SIZE", "25"))
    )
    TRANSLATION_QUEUE_MAX: int = field(
        default_factory=lambda: int(os.getenv("TRANSLATION_QUEUE_MAX", "500"))
    )

    # ---- Firebase --------------------------------------------------------
    FIREBASE_ENABLED: bool = field(
        default_factory=lambda: _bool("FIREBASE_ENABLED", "true")
    )
    FIREBASE_CREDENTIALS_PATH: str = field(
        default_factory=lambda: os.getenv(
            "FIREBASE_CREDENTIALS_PATH", _default_firebase_path()
        )
    )

    # ---- Streaming / VAD -------------------------------------------------
    SAMPLE_RATE: int = 16000
    PARTIAL_INTERVAL_SEC: float = field(
        default_factory=lambda: float(os.getenv("PARTIAL_INTERVAL_SEC", "1.2"))
    )
    SILENCE_DURATION_SEC: float = field(
        default_factory=lambda: float(os.getenv("SILENCE_DURATION_SEC", "0.7"))
    )
    MAX_SEGMENT_SEC: float = field(
        default_factory=lambda: float(os.getenv("MAX_SEGMENT_SEC", "15.0"))
    )
    VAD_ENERGY_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("VAD_ENERGY_THRESHOLD", "0.012"))
    )
    MIN_PARTIAL_AUDIO_SEC: float = field(
        default_factory=lambda: float(os.getenv("MIN_PARTIAL_AUDIO_SEC", "0.3"))
    )

    # ---- Server ----------------------------------------------------------
    HOST: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    CORS_ORIGINS: list[str] = field(
        default_factory=lambda: _list("CORS_ORIGINS", "*")
    )
    MAX_CONCURRENT_TRANSCRIBE_WORKERS: int = field(
        default_factory=lambda: int(os.getenv("MAX_TRANSCRIBE_WORKERS", "4"))
    )
    RELOAD: bool = field(
        default_factory=lambda: _bool("RELOAD", "true")
    )


settings = Settings()