"""
Speech-to-text engines.

Two interchangeable implementations behind one interface:

- FasterWhisperTranscriber: local inference via faster-whisper / CTranslate2.
  This is the default -- it has no per-request network round trip, so it's
  the low-latency path the brief asks for.
- GeminiTranscriber: sends audio to the Google Gemini multimodal API instead.
  Useful as a fallback on machines with no GPU and a slow CPU.

Both are wrapped so the rest of the app never has to know which is active.
"""
from __future__ import annotations

import abc
import asyncio
import io
import logging
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .config import settings

logger = logging.getLogger("transcription")


class TranscriptionResult:
    __slots__ = ("text", "language")

    def __init__(self, text: str, language: str | None = None):
        self.text = text.strip()
        self.language = language


class BaseTranscriber(abc.ABC):
    @abc.abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        """`audio` is mono float32 PCM in [-1, 1] at `sample_rate` Hz."""
        raise NotImplementedError


class FasterWhisperTranscriber(BaseTranscriber):
    """Local, low-latency transcription using CTranslate2 / faster-whisper.

    The model is loaded once (lazily, on first use) and shared by every
    session. WhisperModel.transcribe is safe to call concurrently from
    multiple threads -- each call is stateless w.r.t. the model weights.
    """

    _model = None
    _executor: ThreadPoolExecutor | None = None
    _init_lock = asyncio.Lock()

    def __init__(self):
        self._ensure_executor()

    @classmethod
    def _ensure_executor(cls) -> None:
        if cls._executor is None:
            cls._executor = ThreadPoolExecutor(
                max_workers=settings.MAX_CONCURRENT_TRANSCRIBE_WORKERS,
                thread_name_prefix="whisper",
            )

    async def _ensure_model(self) -> None:
        if FasterWhisperTranscriber._model is not None:
            return
        async with FasterWhisperTranscriber._init_lock:
            if FasterWhisperTranscriber._model is not None:
                return
            from faster_whisper import WhisperModel

            device = settings.WHISPER_DEVICE
            if device == "auto":
                device = self._detect_device()

            logger.info(
                "Loading faster-whisper model=%s device=%s compute_type=%s",
                settings.WHISPER_MODEL_SIZE,
                device,
                settings.WHISPER_COMPUTE_TYPE,
            )
            loop = asyncio.get_event_loop()
            FasterWhisperTranscriber._model = await loop.run_in_executor(
                self._executor,
                lambda: WhisperModel(
                    settings.WHISPER_MODEL_SIZE,
                    device=device,
                    compute_type=settings.WHISPER_COMPUTE_TYPE,
                    local_files_only=True,
                ),
            )
            logger.info("faster-whisper model ready")

    @staticmethod
    def _detect_device() -> str:
        try:
            import ctranslate2
            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        await self._ensure_model()
        loop = asyncio.get_event_loop()

        # Domain bias: pass the configured technical vocabulary as the initial
        # prompt. Whisper uses this as context, which measurably improves
        # accuracy on product names, framework names and jargon.
        initial_prompt = (settings.WHISPER_INITIAL_PROMPT or "").strip() or None

        def _run() -> tuple[str, str | None]:
            segments, info = FasterWhisperTranscriber._model.transcribe(
                audio,
                language=None,
                beam_size=settings.WHISPER_BEAM_SIZE,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                condition_on_previous_text=False,
                initial_prompt=initial_prompt,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            return text, info.language

        text, language = await loop.run_in_executor(self._executor, _run)
        return TranscriptionResult(text=text, language=language)


class GeminiTranscriber(BaseTranscriber):
    """Fallback transcriber that calls the Gemini multimodal API per segment.
    Slower (network round trip per call) but requires no local model weights."""

    def __init__(self):
        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is required to use the Gemini transcription engine")
        import google.generativeai as genai

        genai.configure(api_key=settings.GEMINI_API_KEY)
        self._model = genai.GenerativeModel(settings.GEMINI_MODEL)

    @staticmethod
    def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())
        return buf.getvalue()

    async def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> TranscriptionResult:
        wav_bytes = self._to_wav_bytes(audio, sample_rate)
        loop = asyncio.get_event_loop()

        def _run() -> str:
            response = self._model.generate_content(
                [
                    {"mime_type": "audio/wav", "data": wav_bytes},
                    "Transcribe the speech in this audio clip verbatim, in its "
                    "original language. Return only the transcription text, "
                    "with no commentary or translation.",
                ]
            )
            return (response.text or "").strip()

        text = await loop.run_in_executor(None, _run)
        return TranscriptionResult(text=text, language=None)


def get_transcriber() -> BaseTranscriber:
    """Builds the configured engine. If Gemini is requested but unconfigured,
    fall back to faster-whisper rather than crashing at startup."""
    engine = settings.TRANSCRIPTION_ENGINE.lower()
    if engine == "gemini":
        try:
            return GeminiTranscriber()
        except RuntimeError as exc:
            logger.warning("%s -- falling back to faster-whisper for transcription.", exc)
    return FasterWhisperTranscriber()