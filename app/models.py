"""Plain dataclasses used across the backend. In-memory only -- see the
README's roadmap for what a persistent version would need."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:10]}"


@dataclass
class SessionInfo:
    id: str
    name: str
    target_language: str = "auto"

    # "live" = mic streaming; "upload" = processing an uploaded file
    mode: str = "live"

    # Only set when mode == "upload"
    filename: str | None = None

    # For upload sessions: "ready" -> "processing" -> "done" (or "error").
    # For live sessions: always "ready".
    status: str = "ready"

    # 0.0 to 1.0 -- used for upload progress
    progress: float = 0.0

    # Total duration in seconds (populated from Whisper info for uploads)
    duration_sec: float | None = None

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    listener_count: int = 0
    is_broadcasting: bool = False
    detected_language: str | None = None
    last_partial: str = ""
    last_final_original: str = ""
    last_final_translated: str = ""

    def public_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "target_language": self.target_language,
            "mode": self.mode,
            "filename": self.filename,
            "status": self.status,
            "progress": round(self.progress, 4),
            "duration_sec": self.duration_sec,
            "created_at": self.created_at,
            "listener_count": self.listener_count,
            "is_broadcasting": self.is_broadcasting,
            "detected_language": self.detected_language,
            "last_partial": self.last_partial,
            "last_final_original": self.last_final_original,
            "last_final_translated": self.last_final_translated,
        }