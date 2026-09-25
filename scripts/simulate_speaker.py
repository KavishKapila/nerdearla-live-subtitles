"""
Simulate a live conference speaker by streaming a WAV file to the backend in
real time. Handy for demos and local testing without a live microphone, and
for exercising multiple "simultaneous streams" from one machine (just run
this script several times against different sessions).

The WAV file should be 16-bit PCM; any sample rate / channel count is
resampled/downmixed automatically. Convert other formats with ffmpeg:
    ffmpeg -i input.mp3 -ar 16000 -ac 1 -sample_fmt s16 talk.wav

Usage:
    # create a new session and stream into it
    python scripts/simulate_speaker.py --file talk.wav --create "Keynote"

    # stream into an already-created session
    python scripts/simulate_speaker.py --file talk.wav --session sess_abc123

    # point at a non-default host (e.g. a server on your LAN during the event)
    python scripts/simulate_speaker.py --file talk.wav --create "Keynote" --host 192.168.1.42:8000
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
import wave

import numpy as np

try:
    import websockets
except ImportError:
    print("Missing dependency: pip install websockets")
    sys.exit(1)


def load_wav_as_pcm16(path: str) -> tuple[bytes, int]:
    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"{path} is {sample_width * 8}-bit audio; only 16-bit PCM WAV is supported. "
            "Convert it first, e.g.: ffmpeg -i input.wav -sample_fmt s16 output.wav"
        )

    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    if frame_rate != 16000:
        duration = len(audio) / frame_rate
        target_len = max(1, int(duration * 16000))
        x_old = np.linspace(0, duration, num=len(audio), endpoint=False)
        x_new = np.linspace(0, duration, num=target_len, endpoint=False)
        audio = np.interp(x_new, x_old, audio)

    pcm16 = np.clip(audio, -32768, 32767).astype("<i2")
    return pcm16.tobytes(), 16000


async def stream_audio(ws_url: str, pcm_bytes: bytes, sample_rate: int, chunk_ms: int) -> None:
    chunk_bytes = int(sample_rate * chunk_ms / 1000) * 2  # int16 -> 2 bytes/sample

    async with websockets.connect(ws_url, max_size=None) as ws:
        total_sec = len(pcm_bytes) / 2 / sample_rate
        print(f"Connected to {ws_url}\nStreaming {total_sec:.1f}s of audio in real time (Ctrl+C to stop early)...")
        pos = 0
        start = time.monotonic()
        sent_sec = 0.0
        while pos < len(pcm_bytes):
            chunk = pcm_bytes[pos : pos + chunk_bytes]
            await ws.send(chunk)
            pos += chunk_bytes
            sent_sec += chunk_ms / 1000
            wait = sent_sec - (time.monotonic() - start)
            if wait > 0:
                await asyncio.sleep(wait)
        print("Finished streaming.")


def create_session(api_base: str, name: str, target_language: str) -> str:
    import requests

    resp = requests.post(f"{api_base}/api/sessions", json={"name": name, "target_language": target_language})
    resp.raise_for_status()
    data = resp.json()
    print(f"Created session {data['id']!r} ({data['name']!r}) -- share this id or open the listener view for it.")
    return data["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", required=True, help="Path to a 16-bit PCM WAV file")
    parser.add_argument("--session", help="Existing session id to stream into")
    parser.add_argument("--create", metavar="NAME", help="Create a new session with this name instead of --session")
    parser.add_argument("--target-language", default="auto", choices=["auto", "es", "en"])
    parser.add_argument("--host", default="localhost:8000", help="Backend host:port")
    parser.add_argument("--chunk-ms", type=int, default=250, help="Audio frame size sent per WebSocket message")
    args = parser.parse_args()

    if not args.session and not args.create:
        parser.error("pass either --session <id> or --create <name>")

    api_base = f"http://{args.host}"
    ws_base = f"ws://{args.host}"

    session_id = args.session or create_session(api_base, args.create, args.target_language)
    pcm_bytes, sample_rate = load_wav_as_pcm16(args.file)

    try:
        asyncio.run(stream_audio(f"{ws_base}/ws/audio/{session_id}", pcm_bytes, sample_rate, args.chunk_ms))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
