"""
FastAPI entrypoint for Nerdearla Live Subtitles.

Live:   mic -> /ws/audio/{id}      -> transcribe -> translate -> /ws/subtitles/{id}
Upload: file -> POST /api/sessions/upload -> same pipeline in background

Both produce the same export formats (.srt / .vtt / .json / .txt).
"""
from __future__ import annotations

import json
import logging
import tempfile
import time as _time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import settings
from .session_manager import manager
from . import firebase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("main")

_STARTED_AT = _time.time()

ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpeg", ".mpg",
    ".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".wma",
}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
UPLOAD_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(
        "Nerdearla Vibeathon 2026 -- starting up "
        "(transcription=%s, translation=%s, firebase=%s)",
        settings.TRANSCRIPTION_ENGINE,
        settings.TRANSLATION_ENGINE,
        "on" if settings.FIREBASE_ENABLED else "off",
    )
    firebase_client.init_firebase()
    await manager.warmup()
    yield
    await manager.shutdown()
    logger.info("Backend shut down cleanly")


app = FastAPI(
    title="Nerdearla Live Subtitles",
    description="Real-time multi-session conference speech transcription & translation.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreateSessionRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    target_language: str = Field("auto")


# --------------------------------------------------------------------------- helpers

def _format_srt_time(seconds: float) -> str:
    if seconds < 0:
        seconds = 0.0
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        ms = 0
        secs += 1
        if secs == 60:
            secs = 0
            minutes += 1
            if minutes == 60:
                minutes = 0
                hours += 1
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_vtt_time(seconds: float) -> str:
    return _format_srt_time(seconds).replace(",", ".")


def _cue_lines(seg: dict, lang: str) -> list[str]:
    if lang == "original":
        return [seg["original"]]
    if lang == "translated":
        return [seg.get("translated") or seg["original"]]
    lines = [seg["original"]]
    if seg.get("translated") and seg["translated"] != seg["original"]:
        lines.append(seg["translated"])
    return lines


def _build_srt(history: list[dict], lang: str) -> str:
    out: list[str] = []
    for i, seg in enumerate(history, 1):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", start + 2.0)
        out.append(str(i))
        out.append(f"{_format_srt_time(start)} --> {_format_srt_time(end)}")
        out.extend(_cue_lines(seg, lang))
        out.append("")
    return "\n".join(out)


def _build_vtt(history: list[dict], lang: str) -> str:
    out = ["WEBVTT", ""]
    for i, seg in enumerate(history, 1):
        start = seg.get("start_sec", 0.0)
        end = seg.get("end_sec", start + 2.0)
        out.append(str(i))
        out.append(f"{_format_vtt_time(start)} --> {_format_vtt_time(end)}")
        out.extend(_cue_lines(seg, lang))
        out.append("")
    return "\n".join(out)


# --------------------------------------------------------------------------- REST

@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "transcription_engine": settings.TRANSCRIPTION_ENGINE,
        "translation_engine": settings.TRANSLATION_ENGINE,
        "firebase_ready": firebase_client.is_ready(),
        "active_sessions": len(manager.list_sessions()),
    }


@app.get("/api/metrics")
async def metrics():
    sessions = manager.list_sessions()
    return {
        "uptime_seconds": round(_time.time() - _STARTED_AT, 1),
        "active_sessions": len(sessions),
        "broadcasting_sessions": sum(1 for s in sessions if s["is_broadcasting"]),
        "total_listeners": sum(s["listener_count"] for s in sessions),
        "transcription_engine": settings.TRANSCRIPTION_ENGINE,
        "translation_engine": settings.TRANSLATION_ENGINE,
        "whisper_model": settings.WHISPER_MODEL_SIZE,
        "firebase_ready": firebase_client.is_ready(),
        "sessions": [
            {"id": s["id"], "name": s["name"], "listeners": s["listener_count"],
             "live": s["is_broadcasting"], "mode": s["mode"], "status": s["status"]}
            for s in sessions
        ],
    }


@app.get("/api/sessions")
async def list_sessions():
    return {"sessions": manager.list_sessions()}


@app.get("/api/history")
async def history(limit: int = 20):
    if firebase_client.db is None:
        return {"sessions": [], "firebase": False}
    try:
        docs = (
            firebase_client.db.collection("sessions")
            .order_by("created_at", direction="DESCENDING")
            .limit(limit)
            .stream()
        )
        out = []
        for d in docs:
            data = d.to_dict() or {}
            out.append({
                "id": d.id,
                "name": data.get("name", "(untitled)"),
                "target_language": data.get("target_language", "auto"),
                "mode": data.get("mode", "live"),
                "filename": data.get("filename"),
                "is_broadcasting": data.get("is_broadcasting", False),
                "listener_count": data.get("listener_count", 0),
                "created_at": str(data.get("created_at", "")),
            })
        return {"sessions": out, "firebase": True}
    except Exception:
        logger.exception("Firestore history read failed")
        return {"sessions": [], "firebase": True, "error": "read failed"}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return session.info.public_dict()


@app.post("/api/sessions", status_code=201)
async def create_session(req: CreateSessionRequest):
    session = await manager.create_session(req.name, req.target_language)
    return session.info.public_dict()


@app.post("/api/sessions/upload", status_code=201)
async def upload_session(
    name: str = Form(...),
    target_language: str = Form("auto"),
    file: UploadFile = File(...),
):
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    upload_dir = Path(tempfile.gettempdir()) / "nerdearla-uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_name = f"{uuid.uuid4().hex}{suffix}"
    dest = upload_dir / temp_name

    total = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(UPLOAD_CHUNK)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {MAX_UPLOAD_BYTES // (1024*1024)} MB limit",
                    )
                out.write(chunk)
    except HTTPException:
        raise
    except Exception as exc:
        dest.unlink(missing_ok=True)
        logger.exception("Upload write failed: %s", exc)
        raise HTTPException(status_code=500, detail="Upload failed")

    logger.info("Uploaded %s (%.1f MB) -> %s", filename, total / 1024 / 1024, dest)

    session = await manager.create_upload_session(
        name=name.strip() or filename,
        target_language=target_language,
        file_path=str(dest),
        original_filename=filename,
    )
    return session.info.public_dict()


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    ok = await manager.end_session(session_id)
    if not ok:
        raise HTTPException(status_code=404, detail="session not found")
    return {"ok": True}


# --------------------------------------------------------------------------- exports

@app.get("/api/sessions/{session_id}/subtitles.srt", response_class=Response)
async def subtitles_srt(
    session_id: str,
    lang: Literal["both", "original", "translated"] = Query("both"),
):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    body = _build_srt(session.history, lang)
    safe_name = session.info.name.replace('"', "").replace("\n", " ")
    return Response(
        content=body,
        media_type="application/x-subrip; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.srt"'},
    )


@app.get("/api/sessions/{session_id}/subtitles.vtt", response_class=Response)
async def subtitles_vtt(
    session_id: str,
    lang: Literal["both", "original", "translated"] = Query("both"),
):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    body = _build_vtt(session.history, lang)
    safe_name = session.info.name.replace('"', "").replace("\n", " ")
    return Response(
        content=body,
        media_type="text/vtt; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{safe_name}.vtt"'},
    )


@app.get("/api/sessions/{session_id}/subtitles.json")
async def subtitles_json(session_id: str):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    return {
        "session_id": session.info.id,
        "name": session.info.name,
        "mode": session.info.mode,
        "target_language": session.info.target_language,
        "detected_language": session.info.detected_language,
        "duration_sec": session.info.duration_sec,
        "segments": [
            {
                "index": i,
                "start_sec": seg.get("start_sec", 0.0),
                "end_sec": seg.get("end_sec", 0.0),
                "original": seg["original"],
                "translated": seg.get("translated", ""),
                "source_language": seg.get("source_language", ""),
                "target_language": seg.get("target_language", ""),
            }
            for i, seg in enumerate(session.history, 1)
        ],
    }


@app.get("/api/sessions/{session_id}/transcript.txt", response_class=PlainTextResponse)
async def transcript_txt(session_id: str):
    session = manager.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")

    lines = [
        f"Transcript: {session.info.name}",
        f"Session ID: {session.info.id}",
        f"Mode: {session.info.mode}",
        f"Target language: {session.info.target_language}",
        f"Detected language: {session.info.detected_language or 'unknown'}",
        "", "=" * 70, "",
    ]
    for i, seg in enumerate(session.history, 1):
        lines.append(f"[{i}] ({seg['source_language']}) {seg['original']}")
        if seg.get("translated"):
            lines.append(f"    ({seg['target_language']}) {seg['translated']}")
        lines.append("")
    if not session.history:
        lines.append("(no captions yet)")
    return "\n".join(lines)


# --------------------------------------------------------------------------- WebSockets

@app.websocket("/ws/audio/{session_id}")
async def ws_audio(websocket: WebSocket, session_id: str):
    session = manager.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return
    if session.info.mode != "live":
        await websocket.close(code=4400)
        return
    await websocket.accept()
    session.broadcaster = websocket
    logger.info("Speaker connected -> session %s", session_id)
    try:
        while True:
            try:
                raw = await websocket.receive_bytes()
            except RuntimeError:
                break
            if not raw:
                continue
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
            await session.ingest_audio(samples)
    except WebSocketDisconnect:
        logger.info("Speaker disconnected <- session %s", session_id)
    except Exception:
        logger.exception("Audio socket error for session %s", session_id)
    finally:
        if session.broadcaster is websocket:
            session.broadcaster = None


@app.websocket("/ws/subtitles/{session_id}")
async def ws_subtitles(websocket: WebSocket, session_id: str):
    session = manager.get(session_id)
    if session is None:
        await websocket.close(code=4404)
        return
    await websocket.accept()
    await session.add_subscriber(websocket)
    try:
        await websocket.send_text(json.dumps({
            "type": "meta",
            "session": session.info.public_dict(),
        }))
        # Replay existing history for late joiners of upload sessions
        for idx, seg in enumerate(session.history):
            await websocket.send_text(json.dumps({
                "type": "final",
                "session_id": session.info.id,
                "index": idx,
                "text": seg["original"],
                "source_language": seg["source_language"],
                "start_sec": seg["start_sec"],
                "end_sec": seg["end_sec"],
                "ts": _time.time(),
                "replay": True,
            }))
            if seg.get("translated"):
                await websocket.send_text(json.dumps({
                    "type": "translation",
                    "session_id": session.info.id,
                    "index": idx,
                    "text": seg["translated"],
                    "target_language": seg["target_language"],
                    "start_sec": seg["start_sec"],
                    "end_sec": seg["end_sec"],
                    "ts": _time.time(),
                    "replay": True,
                }))

        while True:
            try:
                await websocket.receive_text()
            except RuntimeError:
                break
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("Subtitles socket error for session %s", session_id)
    finally:
        await session.remove_subscriber(websocket)


# --------------------------------------------------------------------------- static
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
SAMPLES_DIR = BASE_DIR / "samples"

if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")
    logger.info("Serving samples from %s", SAMPLES_DIR)
else:
    logger.warning("No samples/ directory found at %s", SAMPLES_DIR)

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", FRONTEND_DIR)
else:
    logger.warning("No frontend/ directory found at %s", FRONTEND_DIR)