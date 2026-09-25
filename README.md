# Nerdearla Live Subtitles

Real-time multi-session conference transcription & translation.
English ↔ Spanish, WebSocket streaming, downloadable subtitles.

## What it does

- **Live mic sessions** — broadcast a room's audio, audiences see captions on their phones
- **Video/audio upload** — transcribe a recorded talk, export as `.srt` / `.vtt` / `.json`
- **Auto EN↔ES translation** via DeepL
- **Multi-session** — many talks run in parallel, each isolated
- **Firebase Firestore** — sessions and transcripts persisted to the cloud

## Run it

```bash
pip install -r requirements.txt
python run.py
```

Open http://localhost:8000

## Try without a microphone

Click **"▶ Try sample (English)"** or **"▶ Probar ejemplo (Español)"**
in the lobby. Sample audio is bundled in `samples/`.

## Environment

Copy `.env.example` to `.env` and set:

```
DEEPL_API_KEY=your_key_here
```

Optional: `FIREBASE_ENABLED=false` to skip Firestore.

## Architecture

```
mic → WebSocket → faster-whisper → DeepL → WebSocket → browser
                     ↓                ↓
                session buffer    Firestore
```

- `app/main.py` — FastAPI routes + WebSocket handlers
- `app/session_manager.py` — per-session pipelines, VAD, batching
- `app/transcription.py` — faster-whisper wrapper (local, CPU/GPU)
- `app/translation.py` — DeepL REST client
- `frontend/index.html` — single-file UI

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/sessions` | Create a live session |
| `POST /api/sessions/upload` | Upload video/audio |
| `WS /ws/audio/{id}` | Speaker → server (PCM16) |
| `WS /ws/subtitles/{id}` | Server → listeners (JSON) |
| `GET /api/sessions/{id}/subtitles.srt` | Download SRT |
| `GET /api/sessions/{id}/transcript.txt` | Download transcript |

## Scaling

- Each session is an independent asyncio task sharing one Whisper model
- CPU: 2–6 concurrent sessions; GPU: 20+
- Horizontal scale via multiple uvicorn workers behind a reverse proxy,
  with sessions in Redis for cross-process pub/sub

## License

MIT — see LICENSE.