---
title: Reachy Emotion
emoji: 🎭
colorFrom: pink
colorTo: yellow
sdk: static
pinned: false
tags:
  - reachy-mini
  - robotics
  - emotion-detection
  - multimodal
  - gemini
  - conversation
  - voice
---

# Reachy Emotion

Gemini-powered conversation app for Reachy Mini. Talk to Reachy naturally — Gemini maintains the conversation and calls the local emotion-detection SDK as a tool when it wants to know how you're feeling.

## How it works

```
You speak
   ↓
Reachy mic → SpeechRecognition → text → Gemini (with full conversation history)
                                              ↓  when Gemini calls detect_emotion tool
                                   camera + mic → EmotionDetector → result → Gemini
                                              ↓
                                    Gemini response text
                                              ↓
                              RecordedMove (if emotion read) + TTS → Speaker
```

Gemini decides when to read your emotion — it's a tool call, not a continuous loop.

## Prerequisites

- Python 3.10–3.12
- A [Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)
- Reachy Mini robot (or run `--text` mode for testing without hardware)

## Installation

### Option A — `install.sh` (recommended, installs everything)

```bash
git clone https://huggingface.co/spaces/<your-username>/reachy-emotion
cd reachy-emotion
./install.sh
```

This single command:
1. Installs system packages: `ffmpeg` + `portaudio` (via `apt` on Linux, `brew` on macOS)
2. Installs all Python dependencies via `pip install -e .`
3. Creates `.env` from template if it doesn't exist

```bash
./install.sh --dry-run    # preview without making changes
./install.sh --skip-sys   # skip system packages (Python deps only)
```

### Option B — Reachy Mini Dashboard (one-click)

Open Reachy Mini Control at `http://<robot-ip>:8000`, find **Reachy Emotion** in the app store, and click **Install**.

After the dashboard install, SSH into the robot and run once:
```bash
reachy-emotion-setup    # installs ffmpeg + portaudio system packages
```

### Option C — Manual

```bash
pip install -e .           # all Python dependencies
reachy-emotion-setup       # system dependencies (ffmpeg, portaudio)
```

## Configuration

```bash
cp .env.example .env
# Edit .env and set your GEMINI_API_KEY
```

| Variable | Required | Description |
|---|---|---|
| `GEMINI_API_KEY` | Yes | From [aistudio.google.com](https://aistudio.google.com/app/apikey) |
| `GEMINI_MODEL` | No | Default: `gemini-2.5-flash`. Options: `gemini-2.5-flash-lite`, `gemini-2.5-pro` |
| `GEMINI_SYSTEM_PROMPT` | No | Single-line override of Reachy's built-in personality prompt |

## Usage

### From the Reachy Mini Dashboard

After install, **Reachy Emotion** appears in the dashboard. Click **Run** to start — no terminal needed.

### CLI

```bash
reachy-emotion                    # voice mode (speak to Reachy)
reachy-emotion --text             # text mode (type instead of speaking — good for testing)
reachy-emotion --sim              # simulation (start reachy-mini-daemon --sim first)
reachy-emotion --lang fr-FR       # French
```

| Flag | Description |
|---|---|
| `--text` | Type input instead of speaking |
| `--lang CODE` | STT/TTS language, e.g. `en-US`, `fr-FR` (default: `en-US`) |
| `--model NAME` | Override Gemini model |
| `--prompt TEXT` | Override the system prompt for this session only |
| `--sim` | Simulation mode |
| `--media-backend` | Reachy media backend (`default`, `gstreamer`, `webrtc`) |

## Customising Reachy's personality

**Per-run override** (not saved):
```bash
reachy-emotion --prompt "You are Reachy, a robot who speaks only in haiku."
```

**Persistent override** via `.env`:
```ini
GEMINI_SYSTEM_PROMPT=You are Reachy, a friendly robot. Keep all answers under 10 words.
```

**Full edit**: modify `DEFAULT_SYSTEM_PROMPT` in `src/reachy_emotion/gemini_bridge.py`.

Priority: `--prompt` flag > `GEMINI_SYSTEM_PROMPT` env var > built-in default.

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Project Structure

```
reachy-emotion/
├── install.sh                        # Full installer (system + Python deps)
├── .env.example                      # API key template → copy to .env
├── .gitignore
├── index.html                        # HF Space landing page
├── style.css
├── pyproject.toml                    # Package config + all dependencies
├── tests/                            # Unit tests
│   ├── test_gemini_bridge.py
│   ├── test_reachy_handler.py
│   ├── test_tts_announcer.py
│   └── test_voice_input.py
└── src/reachy_emotion/
    ├── main.py                       # ReachyEmotionApp (dashboard) + CLI entry point
    ├── conversation_app.py           # Core conversation loop
    ├── gemini_bridge.py              # Gemini chat session + detect_emotion tool
    ├── voice_input.py                # STT from Reachy mic (energy VAD + Google STT)
    ├── tts_announcer.py              # speak_text() → gTTS + pydub → Reachy speaker
    ├── reachy_handler.py             # RecordedMoves mapping + EMOTIONS_LIBRARY
    └── system_deps.py                # ffmpeg/portaudio check + install
```
