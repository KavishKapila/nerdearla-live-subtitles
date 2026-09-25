"""
Firebase Admin SDK bootstrap + Firestore client.

Kept separate so init runs exactly once and other modules can just import
`db` without worrying about ordering or circular imports.

If `firebase-service-account.json` is missing or invalid, `db` stays None
and every caller gracefully no-ops. The app runs exactly as before.
"""
from __future__ import annotations

import logging
from pathlib import Path

from .config import settings

logger = logging.getLogger("firebase")

db = None            # firestore.Client | None
_initialized = False # idempotency guard


def init_firebase() -> None:
    """Call once at startup. Safe to call multiple times."""
    global db, _initialized
    if _initialized:
        return
    _initialized = True

    if not settings.FIREBASE_ENABLED:
        logger.info("Firebase disabled via FIREBASE_ENABLED=false")
        return

    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        cred_path = Path(settings.FIREBASE_CREDENTIALS_PATH)
        if not cred_path.is_absolute():
            cred_path = (Path(__file__).resolve().parent.parent / cred_path).resolve()
        if not cred_path.exists():
            logger.warning(
                "Firebase credentials not found at %s -- Firestore writes disabled. "
                "Download a service-account key from the Firebase console and save "
                "it at that path, or set FIREBASE_CREDENTIALS_PATH in .env.",
                cred_path,
            )
            return

        if not firebase_admin._apps:
            cred = credentials.Certificate(str(cred_path))
            firebase_admin.initialize_app(cred)

        db = firestore.client()
        logger.info("Firebase Admin SDK ready (project: %s)", cred_path.stem)
    except Exception:
        logger.exception("Firebase init failed -- Firestore writes disabled")
        db = None


def is_ready() -> bool:
    return db is not None