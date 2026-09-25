"""
Translation engine: DeepL REST API.

Zero extra dependencies -- uses `requests` (already in requirements.txt)
and runs the blocking call in a thread pool so it doesn't stall the
asyncio event loop.
"""
from __future__ import annotations

import abc
import asyncio
import logging

import requests

from .config import settings

logger = logging.getLogger("translation")

LANGUAGE_NAMES = {"en": "English", "es": "Spanish"}


# --------------------------------------------------------------------------- base

class BaseTranslator(abc.ABC):
    @abc.abstractmethod
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    async def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [await self.translate(t, source_lang, target_lang) for t in texts]


# --------------------------------------------------------------------------- passthrough

class PassthroughTranslator(BaseTranslator):
    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        return text


# --------------------------------------------------------------------------- DeepL

class DeepLTranslator(BaseTranslator):
    """DeepL REST API via requests. Runs the blocking call in a thread
    executor so it doesn't block the event loop."""

    def __init__(self):
        if not settings.DEEPL_API_KEY:
            raise RuntimeError("DEEPL_API_KEY is required for the DeepL translation engine")
        self._key = settings.DEEPL_API_KEY
        # Free keys end in ":fx", pro keys don't. Pick the right endpoint.
        if self._key.endswith(":fx"):
            self._url = "https://api-free.deepl.com/v2/translate"
        else:
            self._url = "https://api.deepl.com/v2/translate"
        self._headers = {
            "Authorization": f"DeepL-Auth-Key {self._key}",
            "Content-Type": "application/json",
        }
        logger.info("DeepL endpoint: %s", "free" if self._key.endswith(":fx") else "pro")

    def _call_sync(self, texts: list[str], target_lang: str) -> list[str]:
        """Blocking REST call. Runs in a thread pool."""
        payload = {
            "text": texts,
            "target_lang": target_lang.upper(),  # "ES" or "EN"
        }
        r = requests.post(self._url, headers=self._headers, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        return [item["text"].strip() for item in data["translations"]]

    async def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text.strip():
            return ""
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(None, self._call_sync, [text], target_lang)
            return results[0] if results else text
        except Exception:
            logger.exception("DeepL single translation failed; returning original")
            return text

    async def translate_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        if not texts:
            return []

        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            return ["" for _ in texts]

        loop = asyncio.get_event_loop()
        try:
            translated = await loop.run_in_executor(
                None, self._call_sync, [t for _, t in indexed], target_lang
            )
        except Exception:
            logger.exception("DeepL batch failed; falling back to serial")
            translated = []
            for _, t in indexed:
                try:
                    r = await loop.run_in_executor(None, self._call_sync, [t], target_lang)
                    translated.append(r[0] if r else t)
                except Exception:
                    translated.append(t)

        scattered = ["" for _ in texts]
        for pos, (orig_idx, _) in enumerate(indexed):
            scattered[orig_idx] = translated[pos] if pos < len(translated) else texts[orig_idx]
        return scattered


# --------------------------------------------------------------------------- factory

def get_translator() -> BaseTranslator:
    """DeepL only. Passthrough if the key is missing."""
    if settings.DEEPL_API_KEY:
        try:
            translator = DeepLTranslator()
            logger.info("Translation engine: DeepL")
            return translator
        except Exception as exc:
            logger.exception("DeepL init failed: %s", exc)

    logger.warning(
        "DEEPL_API_KEY is not set. Captions will show original language only. "
        "Add DEEPL_API_KEY=... to .env and restart."
    )
    return PassthroughTranslator()