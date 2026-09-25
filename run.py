"""Convenience entrypoint: `python run.py`.

By default shows a clean, friendly startup banner and hides the framework's
INFO logs. Pass `--verbose` (or set VERBOSE=1) to see the full uvicorn /
application logs, which is useful for debugging.
"""
import argparse
import logging
import os
import threading
import time
import urllib.request

import uvicorn

from app.config import settings


# --------------------------------------------------------------------------- helpers

def _print_banner() -> None:
    print()
    print("╔" + "═" * 58 + "╗")
    print("║  🎙️   Nerdearla Live Subtitles" + " " * 26 + "║")
    print("║      Real-time transcription & translation" + " " * 14 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("  Starting up...  (this may take a moment)")
    print("  ▸ First run downloads the Whisper model (~460 MB)")
    print("  ▸ Subsequent runs start in just a few seconds")
    print()


def _probe_health(host: str, port: int) -> bool:
    probe_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    url = f"http://{probe_host}:{port}/api/health"
    try:
        with urllib.request.urlopen(url, timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def _status_reporter(ready_event: threading.Event, stop_event: threading.Event) -> None:
    steps = [
        "Loading configuration...",
        "Initializing Firebase...",
        "Loading Whisper model (the slow part)...",
        "Warming up the transcriber...",
        "Starting web server...",
        "Almost there...",
    ]
    start = time.time()
    step_seconds = 4.0
    printed_idx = -1

    while not stop_event.is_set():
        elapsed = time.time() - start
        idx = min(int(elapsed // step_seconds), len(steps) - 1)
        if idx != printed_idx:
            print(f"   ⏳  {steps[idx]}", flush=True)
            printed_idx = idx
        if ready_event.is_set():
            return
        time.sleep(0.4)


def _watch_health(host: str, port: int,
                  ready_event: threading.Event,
                  stop_event: threading.Event) -> None:
    time.sleep(1.0)
    deadline = time.time() + 600
    while not stop_event.is_set() and time.time() < deadline:
        if _probe_health(host, port):
            ready_event.set()
            return
        time.sleep(0.5)


def _announce_ready(ready_event: threading.Event,
                    stop_event: threading.Event, port: int) -> None:
    ready_event.wait()
    if stop_event.is_set():
        return
    print()
    print("   ✅  Server is ready!")
    print(f"   🌐  Open in your browser:  http://localhost:{port}")
    print(f"   📚  API docs:              http://localhost:{port}/docs")
    print("   🛑  Press Ctrl+C to stop")
    print()


def _configure_logging(verbose: bool) -> None:
    """Silence noisy INFO logs unless --verbose. Errors and warnings always show."""
    if verbose:
        return
    # Uvicorn access + error logs
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    # App loggers
    for name in ("main", "firebase", "session", "transcription", "translation", "vibeathon"):
        logging.getLogger(name).setLevel(logging.WARNING)
    # Third-party noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("deep_translator").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- main

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Nerdearla Live Subtitles server.")
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=os.getenv("VERBOSE", "0") not in ("0", "", "false", "False"),
        help="Show full uvicorn and application logs (default: quiet mode).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(verbose=args.verbose)

    _print_banner()
    if args.verbose:
        print("   (verbose mode: full logs enabled)")
        print()

    ready_event = threading.Event()
    stop_event = threading.Event()

    threading.Thread(target=_status_reporter, args=(ready_event, stop_event), daemon=True).start()
    threading.Thread(target=_watch_health, args=(settings.HOST, settings.PORT, ready_event, stop_event), daemon=True).start()
    threading.Thread(target=_announce_ready, args=(ready_event, stop_event, settings.PORT), daemon=True).start()

    try:
        uvicorn.run(
            "app.main:app",
            host=settings.HOST,
            port=settings.PORT,
            reload=settings.RELOAD,
            log_level="info" if args.verbose else "warning",
        )
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        ready_event.set()
        print("\n   👋  Shut down cleanly.")


if __name__ == "__main__":
    main()