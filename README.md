# Reachy Emotion

A Gemini-powered conversation app for [Reachy Mini](https://pollen-robotics.com/reachy-mini). Talk to Reachy naturally — Gemini maintains the conversation and calls a cloud emotion detection service as a tool when it wants to read how you're feeling.

Emotion inference runs on **emotion-cloud**: a Two-Tower Multimodal Transformer (ViT-B/16 + emotion2vec) deployed on Google Kubernetes Engine. Camera frames stream continuously to the cloud in the background; Gemini decides *when* to look at the latest result.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Reachy Mini Robot                          │
│                                                                     │
│   Microphone ──► SpeechRecognition ──► text ──► GeminiBridge        │
│                                                      │              │
│                                           detect_emotion tool       │
│                                                      │              │
│                                         EmotionCloudClient          │
│                                         (latest result store)       │
│                                                      │              │
│   Camera ──────────────────────────────► background gRPC stream     │
│   Microphone (optional) ───────────────►  (15 fps, bidirectional)   │
│                                                      │              │
│                                         Gemini response text        │
│                                                      │              │
│   Speaker ◄── TTS (gTTS + pydub) ◄──────────────────┤              │
│   Antennas/Body ◄── RecordedMoves ◄─────────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                │ gRPC (bidirectional stream)
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  emotion-cloud  (Google Kubernetes Engine)           │
│                                                                     │
│   gRPC server (port 50051)                                          │
│        │                                                            │
│   TorchServe ──► Two-Tower Transformer                              │
│                   ├── Video: ViT-B/16 (AffectNet, 450 K faces)      │
│                   └── Audio: emotion2vec_base                       │
│                        ↓ bidirectional cross-attention              │
│                   EmotionResponse                                   │
│                   ├── dominant_emotion (8 classes)                  │
│                   ├── confidence_scores                             │
│                   └── stress / engagement / arousal                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design:** Gemini decides *when* to read emotion — it's an on-demand tool call, not a continuous loop. But the camera stream to emotion-cloud runs continuously in the background so the answer is always fresh when the tool is called.

### Code structure

```
src/reachy_emotion/
├── main.py               ← ReachyEmotionApp (dashboard entry point) + CLI
├── conversation_app.py   ← core conversation loop
├── gemini_bridge.py      ← Gemini chat session + detect_emotion tool
├── cloud_client.py       ← gRPC streaming client for emotion-cloud
├── proto/
│   └── emotion.proto     ← gRPC service definition (stubs auto-generated)
├── voice_input.py        ← STT from Reachy mic (energy VAD + Google STT)
├── tts_announcer.py      ← speak_text() → gTTS + pydub → Reachy speaker
├── reachy_handler.py     ← ActionCommand → RecordedMoves / Motion
└── system_deps.py        ← ffmpeg/portaudio check + install
```

---

## Prerequisites

- Python 3.10–3.12
- Reachy Mini robot (or `--text` / `--sim` for testing without hardware)
- A Gemini API key — free tier at [aistudio.google.com](https://aistudio.google.com/app/apikey)
- **emotion-cloud** deployed and reachable — see the [emotion-cloud repo](https://github.com/saurabh947/emotion-cloud) for GKE setup
- System packages: `ffmpeg` (TTS) and `portaudio` (microphone)

---

## Installation

### Option A — `install.sh` (recommended)

```bash
git clone https://github.com/<your-username>/reachy-emotion
cd reachy-emotion
./install.sh
```

This single command installs system packages (`ffmpeg` + `portaudio`), all Python dependencies (including `grpcio` and `grpcio-tools`), and creates a `.env` template.

```bash
./install.sh --dry-run    # preview without making changes
./install.sh --skip-sys   # skip system packages (Python deps only)
```

### Option B — Reachy Mini Dashboard

Open Reachy Mini Control at `http://<robot-ip>:8000`, find **Reachy Emotion** in the app store, and click **Install**. Then SSH into the robot and run:

```bash
reachy-emotion-setup    # installs ffmpeg + portaudio
```

### Option C — Manual

```bash
pip install -e .        # all Python dependencies
reachy-emotion-setup    # system dependencies (ffmpeg, portaudio)
```

> **Note:** gRPC stubs (`emotion_pb2.py`, `emotion_pb2_grpc.py`) are generated automatically from `proto/emotion.proto` on first run using `grpcio-tools`. No manual `protoc` step needed.

---

## Configuration

```bash
cp .env.example .env
# Edit .env — set GEMINI_API_KEY and EMOTION_CLOUD_ENDPOINT
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | From [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `EMOTION_CLOUD_ENDPOINT` | Yes | GKE external IP + port, e.g. `34.x.x.x:50051` |
| `GEMINI_MODEL` | No | Default: `gemini-2.5-flash` |
| `GEMINI_SYSTEM_PROMPT` | No | Single-line override of Reachy's personality prompt |

### Finding the emotion-cloud endpoint

```bash
# From the emotion-cloud repo, after deploying to GKE:
kubectl get svc emotion-cloud-grpc -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
# Returns something like: 34.102.x.x
# Use as: EMOTION_CLOUD_ENDPOINT=34.102.x.x:50051
```

---

## Starting the daemon

The Reachy daemon must be running before starting the app.

**Linux / on the robot:**
```bash
reachy-mini-daemon
```

**macOS (Lite / USB connection):**
```bash
# mjpython is required on macOS for MuJoCo compatibility
mjpython -m reachy_mini.daemon.app.main
```

**Simulation (no robot):**
```bash
# Linux:
reachy-mini-daemon --sim
# macOS:
mjpython -m reachy_mini.daemon.app.main --sim
```

Verify at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Usage

```bash
reachy-emotion                                        # voice mode
reachy-emotion --text                                 # text mode
reachy-emotion --sim                                  # simulation mode
reachy-emotion --lang fr-FR                           # French
reachy-emotion --cloud-endpoint 34.x.x.x:50051       # override endpoint
```

| Flag | Description |
|---|---|
| `--text` | Type input instead of speaking |
| `--lang CODE` | STT/TTS language, e.g. `en-US`, `fr-FR` (default: `en-US`) |
| `--model NAME` | Override Gemini model |
| `--prompt TEXT` | Override system prompt for this session |
| `--sim` | Simulation mode |
| `--media-backend` | Reachy media backend: `default`, `gstreamer`, `webrtc` |
| `--cloud-endpoint` | emotion-cloud gRPC address (overrides `EMOTION_CLOUD_ENDPOINT`) |

Or launch from the **Reachy Mini Dashboard** — no terminal needed.

---

## Customising Reachy's personality

**Per-run** (not saved):
```bash
reachy-emotion --prompt "You are Reachy, a robot who speaks only in haiku."
```

**Persistent** via `.env`:
```ini
GEMINI_SYSTEM_PROMPT=You are Reachy, a friendly robot. Keep answers under 10 words.
```

**Full edit**: modify `DEFAULT_SYSTEM_PROMPT` in `src/reachy_emotion/gemini_bridge.py`.

Priority: `--prompt` flag > `GEMINI_SYSTEM_PROMPT` env var > built-in default.

---

## Running tests

Requires Python 3.10–3.12 (the package requires `>=3.10`).

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The unit tests stub out all robot hardware, the Gemini API, and gRPC — no robot, no internet connection, and no running emotion-cloud instance is needed.

---

## Troubleshooting

### `EMOTION_CLOUD_ENDPOINT is not set`
Add the GKE LoadBalancer IP to your `.env`:
```bash
# Get the IP from GKE:
kubectl get svc emotion-cloud-grpc -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
# Then set in .env:
EMOTION_CLOUD_ENDPOINT=34.x.x.x:50051
```

### `emotion-cloud health check failed`
The app logs a warning and continues — emotion detection will return "unclear" until the cloud is reachable. Check:
- emotion-cloud is deployed: `kubectl get pods`
- The service has an external IP: `kubectl get svc emotion-cloud-grpc`
- Port 50051 is reachable from the robot's network

### `ffmpeg not found`
Required for TTS (gTTS outputs MP3, robot speaker needs WAV).
```bash
reachy-emotion-setup   # auto-detects OS and installs ffmpeg + portaudio
```

### `ModuleNotFoundError: No module named 'reachy_mini'`
Install the Reachy Mini SDK or run in simulation mode:
```bash
reachy-emotion --sim --text
```

### gRPC stubs not generating
The stubs (`emotion_pb2.py`, `emotion_pb2_grpc.py`) are auto-generated from `proto/emotion.proto` on first run. If generation fails:
```bash
pip install grpcio-tools
# Then run the app — stubs are generated automatically at startup
```

---

## License

MIT
