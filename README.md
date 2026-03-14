# Reachy Emotion

A Gemini-powered conversation app for [Reachy Mini](https://pollen-robotics.com/reachy-mini). Talk to Reachy naturally — Gemini maintains the conversation and calls the local [emotion-detection-action](https://github.com/saurabh947/emotion-detection-action) SDK as a tool when it wants to read how you're feeling.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Reachy Mini Robot                       │
│                                                                 │
│   Microphone ──► SpeechRecognition ──► text ──► GeminiBridge   │
│                                                      │          │
│                                           detect_emotion tool   │
│                                                      │          │
│   Camera ──────────────────────────────► EmotionDetector        │
│   Microphone ──────────────────────────►  (emotion-detection-   │
│                                            action SDK)          │
│                                                      │          │
│                                           NeuralEmotionResult   │
│                                                      │          │
│                                         Gemini response text    │
│                                                      │          │
│   Speaker ◄── TTS (gTTS + pydub) ◄──────────────────┤          │
│   Antennas/Body ◄── RecordedMoves ◄─────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

Gemini decides *when* to read emotion — it's an on-demand tool call, not a continuous loop.

### Code structure

```
src/reachy_emotion/
├── main.py               ← ReachyEmotionApp (dashboard entry point) + CLI
├── conversation_app.py   ← core conversation loop
├── gemini_bridge.py      ← Gemini chat session + detect_emotion tool
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
- System packages: `ffmpeg` (TTS) and `portaudio` (microphone)

---

## Installation

### Option A — `install.sh` (recommended)

```bash
git clone https://github.com/<your-username>/reachy-emotion
cd reachy-emotion
./install.sh
```

This single command installs system packages (`ffmpeg` + `portaudio`), all Python dependencies, and creates a `.env` template.

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

---

## Configuration

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | From [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | No | Default: `gemini-2.5-flash` |
| `GEMINI_SYSTEM_PROMPT` | No | Single-line override of Reachy's personality prompt |

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
reachy-emotion              # voice mode (speak to Reachy)
reachy-emotion --text       # text mode (type instead of speaking)
reachy-emotion --sim        # simulation mode (daemon must run with --sim)
reachy-emotion --lang fr-FR # French
```

| Flag | Description |
|---|---|
| `--text` | Type input instead of speaking |
| `--lang CODE` | STT/TTS language, e.g. `en-US`, `fr-FR` (default: `en-US`) |
| `--model NAME` | Override Gemini model |
| `--prompt TEXT` | Override system prompt for this session |
| `--sim` | Simulation mode |
| `--media-backend` | Reachy media backend: `default`, `gstreamer`, `webrtc` |

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

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
