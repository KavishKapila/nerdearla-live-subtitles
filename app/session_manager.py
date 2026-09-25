"""
One ConferenceSession per talk. Two modes:

  live:   audio buffer -> transcribe -> translate -> broadcast (real time)
  upload: a video/audio file -> segment-by-segment transcription -> broadcast

Upload mode decouples transcription from translation: the transcriber races
ahead and the translation queue drains in parallel batches, which is ~6-10x
faster than the naive serial pipeline.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time

import numpy as np
from fastapi import WebSocket

from .config import settings
from .models import SessionInfo, new_id
from .transcription import BaseTranscriber, get_transcriber
from .translation import BaseTranslator, get_translator
from . import firebase_client

logger = logging.getLogger("session")


class ConferenceSession:
    def __init__(
        self,
        name: str,
        target_language: str,
        transcriber: BaseTranscriber,
        translator: BaseTranslator,
        mode: str = "live",
        file_path: str | None = None,
        original_filename: str | None = None,
    ):
        self.info = SessionInfo(
            id=new_id("sess_"),
            name=name,
            target_language=target_language,
            mode=mode,
            filename=original_filename,
        )
        self.transcriber = transcriber
        self.translator = translator

        # live-mode state
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_lock = asyncio.Lock()
        self._has_speech = False
        self._last_speech_time = 0.0
        self._last_partial_emit_time = 0.0
        self._segment_start_time = 0.0
        self._segment_started_at: float | None = None

        # upload-mode state
        self._file_path = file_path
        self._translation_queue: asyncio.Queue | None = None
        self._translation_worker_task: asyncio.Task | None = None

        self.subscribers: set[WebSocket] = set()
        self.broadcaster: WebSocket | None = None
        self._processing_task: asyncio.Task | None = None
        self._closed = False

        self.history: list[dict] = []
        self._session_started_at = time.monotonic()
        self._firestore_ref = None

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        self.info.is_broadcasting = True
        self._session_started_at = time.monotonic()
        self._segment_start_time = time.monotonic()

        if self.info.mode == "upload":
            self.info.status = "processing"
            self._processing_task = asyncio.create_task(self._upload_processing_loop())
        else:
            self._processing_task = asyncio.create_task(self._live_processing_loop())

        self._create_firestore_doc()

    async def stop(self) -> None:
        self._closed = True
        self.info.is_broadcasting = False
        if self._processing_task:
            self._processing_task.cancel()
        if self._translation_worker_task:
            self._translation_worker_task.cancel()
        for ws in list(self.subscribers):
            try:
                await ws.close(code=4000)
            except Exception:
                pass
        self.subscribers.clear()
        self._close_firestore_doc()
        self._cleanup_upload_file()

    def _cleanup_upload_file(self) -> None:
        if self.info.mode != "upload" or not self._file_path:
            return
        try:
            if os.path.exists(self._file_path):
                os.unlink(self._file_path)
                logger.info("Cleaned up upload temp file: %s", self._file_path)
        except Exception:
            logger.warning("Could not remove temp file %s", self._file_path)

    # ------------------------------------------------------------------ Firestore

    def _create_firestore_doc(self) -> None:
        if firebase_client.db is None:
            return
        try:
            from firebase_admin import firestore as fs
            self._firestore_ref = firebase_client.db.collection("sessions").document(self.info.id)
            self._firestore_ref.set({
                "name": self.info.name,
                "target_language": self.info.target_language,
                "mode": self.info.mode,
                "filename": self.info.filename,
                "status": self.info.status,
                "created_at": fs.SERVER_TIMESTAMP,
                "is_broadcasting": True,
                "listener_count": 0,
            })
            logger.info("Firestore: created session doc %s", self.info.id)
        except Exception:
            logger.exception("Firestore session create failed (non-fatal)")
            self._firestore_ref = None

    def _close_firestore_doc(self) -> None:
        if self._firestore_ref is None:
            return
        try:
            from firebase_admin import firestore as fs
            self._firestore_ref.update({
                "is_broadcasting": False,
                "status": self.info.status,
                "ended_at": fs.SERVER_TIMESTAMP,
            })
        except Exception:
            logger.exception("Firestore session close failed (non-fatal)")

    def _persist_segment(self, segment: dict) -> None:
        if self._firestore_ref is None:
            return
        try:
            from firebase_admin import firestore as fs
            self._firestore_ref.collection("transcripts").add({
                "original": segment["original"],
                "source_language": segment["source_language"],
                "translated": segment["translated"],
                "target_language": segment["target_language"],
                "start_sec": segment["start_sec"],
                "end_sec": segment["end_sec"],
                "timestamp": fs.SERVER_TIMESTAMP,
            })
        except Exception:
            logger.exception("Firestore transcript write failed (non-fatal)")

    # ------------------------------------------------------------------ audio in (live only)

    async def ingest_audio(self, chunk: np.ndarray) -> None:
        if self.info.mode != "live":
            return
        if chunk.size == 0:
            return
        now = time.monotonic()
        energy = float(np.sqrt(np.mean(np.square(chunk))))
        async with self._buffer_lock:
            self._buffer = np.concatenate([self._buffer, chunk])
            if energy >= settings.VAD_ENERGY_THRESHOLD:
                if self._segment_started_at is None:
                    self._segment_started_at = now
                self._has_speech = True
                self._last_speech_time = now

    # ------------------------------------------------------------ subscribers out

    async def add_subscriber(self, ws: WebSocket) -> None:
        self.subscribers.add(ws)
        self.info.listener_count = len(self.subscribers)
        await self._broadcast_meta()
        if self._firestore_ref is not None:
            try:
                self._firestore_ref.update({"listener_count": self.info.listener_count})
            except Exception:
                pass

    async def remove_subscriber(self, ws: WebSocket) -> None:
        self.subscribers.discard(ws)
        self.info.listener_count = len(self.subscribers)

    async def _broadcast(self, message: dict) -> None:
        if not self.subscribers:
            return
        payload = json.dumps(message)
        dead = []
        for ws in self.subscribers:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.subscribers.discard(ws)
        if dead:
            self.info.listener_count = len(self.subscribers)

    async def _broadcast_meta(self) -> None:
        await self._broadcast({"type": "meta", "session": self.info.public_dict()})

    # ==================================================================
    # LIVE MODE pipeline (unchanged -- segments are naturally spaced out)
    # ==================================================================

    async def _live_processing_loop(self) -> None:
        check_interval = 0.15
        try:
            while not self._closed:
                await asyncio.sleep(check_interval)
                now = time.monotonic()
                async with self._buffer_lock:
                    buf_len = len(self._buffer)
                    has_speech = self._has_speech
                    silence_elapsed = (now - self._last_speech_time) if has_speech else 0.0
                    segment_elapsed = now - self._segment_start_time

                if buf_len == 0 or not has_speech:
                    continue

                should_finalize = (
                    silence_elapsed >= settings.SILENCE_DURATION_SEC
                    or segment_elapsed >= settings.MAX_SEGMENT_SEC
                )
                if should_finalize:
                    await self._finalize_live_segment()
                elif now - self._last_partial_emit_time >= settings.PARTIAL_INTERVAL_SEC:
                    self._last_partial_emit_time = now
                    await self._emit_partial()
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Live processing loop crashed for %s", self.info.id)

    async def _emit_partial(self) -> None:
        async with self._buffer_lock:
            snapshot = self._buffer.copy()
        if len(snapshot) < settings.SAMPLE_RATE * settings.MIN_PARTIAL_AUDIO_SEC:
            return
        try:
            result = await self.transcriber.transcribe(snapshot, settings.SAMPLE_RATE)
        except Exception:
            logger.exception("Partial transcription failed for %s", self.info.id)
            return
        if not result.text:
            return
        self.info.last_partial = result.text
        await self._broadcast({
            "type": "partial",
            "session_id": self.info.id,
            "text": result.text,
            "ts": time.time(),
        })

    async def _finalize_live_segment(self) -> None:
        async with self._buffer_lock:
            snapshot = self._buffer.copy()
            snapshot_len = len(snapshot)
            seg_started = self._segment_started_at

        try:
            result = await self.transcriber.transcribe(snapshot, settings.SAMPLE_RATE)
        except Exception:
            logger.exception("Final transcription failed for %s", self.info.id)
            result = None

        now_mono = time.monotonic()
        end_sec = max(0.0, now_mono - self._session_started_at)
        start_sec = max(0.0, (seg_started or now_mono) - self._session_started_at)

        async with self._buffer_lock:
            self._buffer = self._buffer[snapshot_len:]
            still_recent = (time.monotonic() - self._last_speech_time) < settings.SILENCE_DURATION_SEC
            self._has_speech = len(self._buffer) > 0 and still_recent
            self._segment_start_time = time.monotonic()
            self._last_partial_emit_time = 0.0
            self._segment_started_at = None

        if result is None or not result.text:
            return

        source_lang = (result.language or "en")[:2].lower()
        self.info.detected_language = source_lang
        self.info.last_final_original = result.text
        self.info.last_partial = ""

        await self._broadcast({
            "type": "final",
            "session_id": self.info.id,
            "text": result.text,
            "source_language": source_lang,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "ts": time.time(),
        })

        target_lang = self._resolve_target_language(source_lang)
        translated = await self.translator.translate(
            result.text, source_lang=source_lang, target_lang=target_lang
        )
        self.info.last_final_translated = translated

        await self._broadcast({
            "type": "translation",
            "session_id": self.info.id,
            "text": translated,
            "target_language": target_lang,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
            "ts": time.time(),
        })

        segment = {
            "original": result.text,
            "source_language": source_lang,
            "translated": translated,
            "target_language": target_lang,
            "start_sec": round(start_sec, 3),
            "end_sec": round(end_sec, 3),
        }
        self.history.append(segment)
        self._persist_segment(segment)

    # ==================================================================
    # UPLOAD MODE pipeline (parallel translation)
    # ==================================================================

    async def _upload_processing_loop(self) -> None:
        """Transcribe an uploaded file, streaming originals immediately and
        translating in parallel batches behind the scenes.

        Ordering is preserved: batches are dispatched in segment order and
        broadcast in order, so translations always appear under the correct
        caption even though the actual HTTP calls race."""
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        # Start the translation worker
        self._translation_queue = asyncio.Queue(maxsize=settings.TRANSLATION_QUEUE_MAX)
        self._translation_worker_task = asyncio.create_task(self._translation_worker())

        def _producer():
            try:
                model = self.transcriber._model  # type: ignore[attr-defined]
                if model is None:
                    asyncio.run_coroutine_threadsafe(
                        self.transcriber.transcribe(np.zeros(1, dtype=np.float32), 16000),
                        loop,
                    ).result()
                    model = self.transcriber._model  # type: ignore[attr-defined]

                segments, info = model.transcribe(
                    self._file_path,
                    beam_size=settings.WHISPER_BEAM_SIZE,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=300),
                    condition_on_previous_text=False,
                    initial_prompt=(settings.WHISPER_INITIAL_PROMPT or "").strip() or None,
                )
                asyncio.run_coroutine_threadsafe(queue.put(("info", info)), loop)
                for seg in segments:
                    asyncio.run_coroutine_threadsafe(queue.put(("segment", seg)), loop)
                asyncio.run_coroutine_threadsafe(queue.put(("done", None)), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(queue.put(("error", e)), loop)

        loop.run_in_executor(None, _producer)

        segment_index = 0
        last_progress_broadcast = 0.0

        while True:
            if self._closed:
                break
            kind, payload = await queue.get()

            if kind == "done":
                break
            if kind == "error":
                self.info.status = "error"
                logger.exception("Upload processing failed: %s", payload)
                await self._broadcast({
                    "type": "error",
                    "session_id": self.info.id,
                    "message": str(payload),
                })
                break
            if kind == "info":
                self.info.duration_sec = float(getattr(payload, "duration", 0.0) or 0.0)
                self.info.detected_language = (getattr(payload, "language", None) or "en")[:2].lower()
                await self._broadcast_meta()
                continue

            # kind == "segment"
            seg = payload
            text = (seg.text or "").strip()
            if not text:
                continue

            segment_index += 1
            start_sec = float(seg.start)
            end_sec = float(seg.end)

            source_lang = (self.info.detected_language or "en")[:2].lower()
            target_lang = self._resolve_target_language(source_lang)

            # Broadcast original immediately -- user sees it as soon as it's decoded
            await self._broadcast({
                "type": "final",
                "session_id": self.info.id,
                "text": text,
                "source_language": source_lang,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "ts": time.time(),
            })

            # Pre-allocate the history slot (keeps export order correct)
            index = len(self.history)
            self.history.append({
                "original": text,
                "source_language": source_lang,
                "translated": "",
                "target_language": target_lang,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
            })

            # Queue the translation (non-blocking unless queue is full)
            try:
                await asyncio.wait_for(
                    self._translation_queue.put({
                        "index": index,
                        "text": text,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "start_sec": start_sec,
                        "end_sec": end_sec,
                    }),
                    timeout=30.0,
                )
            except asyncio.TimeoutError:
                logger.warning("Translation queue backed up for %s", self.info.id)

            if self.info.duration_sec:
                self.info.progress = min(1.0, end_sec / self.info.duration_sec)
            now = time.time()
            if now - last_progress_broadcast > 1.0:
                last_progress_broadcast = now
                await self._broadcast({
                    "type": "progress",
                    "session_id": self.info.id,
                    "progress": self.info.progress,
                    "segments_done": segment_index,
                })

        # Signal translation worker to finish and wait for the queue to drain
        if self._translation_queue is not None:
            await self._translation_queue.put(None)  # sentinel
        if self._translation_worker_task:
            try:
                await asyncio.wait_for(self._translation_worker_task, timeout=600.0)
            except asyncio.TimeoutError:
                logger.warning("Translation worker didn't finish in time for %s", self.info.id)
                self._translation_worker_task.cancel()

        # Final state
        if self.info.status != "error":
            self.info.status = "done"
            self.info.progress = 1.0
        self.info.is_broadcasting = False
        self._cleanup_upload_file()

        await self._broadcast({
            "type": "progress",
            "session_id": self.info.id,
            "progress": 1.0,
            "segments_done": segment_index,
            "done": True,
        })
        await self._broadcast_meta()

    async def _translation_worker(self) -> None:
        """Drains the translation queue in parallel batches. Each batch of
        up to N segments is translated concurrently with asyncio.gather,
        then the results are broadcast in the original order so the UI
        always shows the translation under the correct caption."""
        batch_size = max(1, settings.TRANSLATION_BATCH_SIZE)
        try:
            while True:
                try:
                    first = await self._translation_queue.get()
                except asyncio.CancelledError:
                    return
                except Exception:
                    logger.exception("Translation worker queue.get failed")
                    await asyncio.sleep(0.5)
                    continue

                if first is None:
                    return  # clean shutdown

                batch = [first]

                # Drain up to batch_size more (non-blocking)
                while len(batch) < batch_size:
                    try:
                        nxt = self._translation_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if nxt is None:
                        # Sentinel arrived -- put it back so the loop sees it
                        await self._translation_queue.put(None)
                        break
                    batch.append(nxt)

                await self._translate_batch(batch)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Translation worker crashed for %s", self.info.id)

    async def _translate_batch(self, batch: list[dict]) -> None:
        """Translate a batch in parallel, then broadcast the results in order."""

        async def _one(item: dict):
            try:
                return await self.translator.translate(
                    item["text"],
                    source_lang=item["source_lang"],
                    target_lang=item["target_lang"],
                )
            except Exception:
                logger.exception("Translation failed for one item")
                return item["text"]  # fall back to original

        results = await asyncio.gather(*(_one(x) for x in batch))

        for item, translated in zip(batch, results):
            # Fill in the pre-allocated history slot
            idx = item["index"]
            if 0 <= idx < len(self.history):
                self.history[idx]["translated"] = translated

            await self._broadcast({
                "type": "translation",
                "session_id": self.info.id,
                "text": translated,
                "target_language": item["target_lang"],
                "start_sec": round(item["start_sec"], 3),
                "end_sec": round(item["end_sec"], 3),
                "ts": time.time(),
            })

            self._persist_segment({
                "original": item["text"],
                "source_language": item["source_lang"],
                "translated": translated,
                "target_language": item["target_lang"],
                "start_sec": round(item["start_sec"], 3),
                "end_sec": round(item["end_sec"], 3),
            })

    # ------------------------------------------------------------------ helpers

    def _resolve_target_language(self, source_lang: str) -> str:
        if self.info.target_language in ("es", "en"):
            return self.info.target_language
        return "en" if source_lang == "es" else "es"


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, ConferenceSession] = {}
        self._lock = asyncio.Lock()
        self._transcriber = get_transcriber()
        self._translator = get_translator()

    async def warmup(self) -> None:
        try:
            silence = np.zeros(settings.SAMPLE_RATE, dtype=np.float32)
            await self._transcriber.transcribe(silence, settings.SAMPLE_RATE)
            logger.info("Transcriber warmed up.")
        except Exception:
            logger.exception("Transcriber warmup failed (non-fatal)")

    async def create_session(self, name: str, target_language: str = "auto") -> ConferenceSession:
        session = ConferenceSession(name, target_language, self._transcriber, self._translator, mode="live")
        async with self._lock:
            self._sessions[session.info.id] = session
        session.start()
        logger.info("Created LIVE session %s (%s)", session.info.id, name)
        return session

    async def create_upload_session(
        self,
        name: str,
        target_language: str,
        file_path: str,
        original_filename: str,
    ) -> ConferenceSession:
        session = ConferenceSession(
            name=name,
            target_language=target_language,
            transcriber=self._transcriber,
            translator=self._translator,
            mode="upload",
            file_path=file_path,
            original_filename=original_filename,
        )
        async with self._lock:
            self._sessions[session.info.id] = session
        session.start()
        logger.info("Created UPLOAD session %s (%s -> %s)",
                    session.info.id, original_filename, name)
        return session

    def get(self, session_id: str) -> ConferenceSession | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[dict]:
        return [s.info.public_dict() for s in self._sessions.values()]

    async def end_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if not session:
            return False
        await session.stop()
        logger.info("Ended session %s", session_id)
        return True

    async def shutdown(self) -> None:
        for session_id in list(self._sessions.keys()):
            await self.end_session(session_id)


manager = SessionManager()