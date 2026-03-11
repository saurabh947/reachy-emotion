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

Lightweight Reachy Mini app that detects human emotions using the robot's built-in camera and microphone, then responds with pre-recorded robot movements and spoken announcements.

## Architecture

This app ships two modes, both registered as separate Reachy Mini dashboard apps:

### Mode 1 — Continuous Emotion Detection Loop (`ReachyEmotionApp`)

```
Camera + Mic → EmotionDetector → ActionCommand → ReachyMiniActionHandler
                                                      ↓              ↓
                                               RecordedMoves    speak_text (TTS)
                                                      ↓              ↓
                                                  Motion          Speaker
```

Detects emotions on every camera frame, plays matching pre-recorded animations and announces the emotion aloud.

### Mode 2 — Gemini Conversation (`ReachyConversationApp`)

```
Mic → SpeechRecognition STT → Gemini Chat Session (with history)
                                        ↓ (tool call when needed)
                               detect_emotion → Camera + Mic → EmotionDetector
                                        ↓
                               response text → speak_text (TTS) → Speaker
                               emotion result → RecordedMoves
```

Gemini drives the conversation. The `detect_emotion` function is registered as a Gemini tool call — Gemini decides when to read the emotion, then uses the result to inform its reply. Full conversation history is maintained automatically by the `google-genai` SDK's `Chat` object.

## Installation

### Via Reachy Mini Dashboard (recommended)

Open Reachy Mini Control (`http://<robot-ip>:8000`), find **Reachy Emotion** in the app store, and click **Install**. All dependencies are installed automatically.

### Manual install

```bash
pip install git+https://github.com/pollen-robotics/reachy_mini.git  # if not already installed

# Clone and install this app
git clone https://huggingface.co/spaces/<your-username>/reachy-emotion
cd reachy-emotion
pip install -e .
```

All Python dependencies (including the `emotion-detection-action` SDK) are pulled automatically by `pip install -e .`.

> **System dependency — ffmpeg**: Required for TTS (speech announcements via speaker).
> Without it the app still runs; only voice announcements are skipped.
> ```bash
> brew install ffmpeg        # macOS
> sudo apt install ffmpeg    # Ubuntu/Debian
> ```

## Configuration

Copy `.env.example` to `.env` and fill in your Gemini API key:

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

Get a free API key at [aistudio.google.com](https://aistudio.google.com/app/apikey).

## Usage

### From the Dashboard

Two apps appear in Reachy Mini Control after installation:

| App | Description |
|---|---|
| **Reachy Emotion** | Continuous emotion detection loop — detects faces, plays moves, announces emotions |
| **Reachy Conversation** | Gemini-powered voice chat with emotion detection as a tool call |

### CLI — Emotion detection loop

```bash
reachy-emotion                       # Real robot
reachy-emotion --sim                 # Simulation
reachy-emotion --no-robot            # Webcam + logging only (no hardware)
```

### CLI — Gemini conversation

```bash
reachy-emotion-chat                  # Voice mode (speak to Reachy)
reachy-emotion-chat --text           # Text mode (type to Reachy)
reachy-emotion-chat --sim            # Simulation
reachy-emotion-chat --lang fr-FR     # French voice input + responses
```

### Options — emotion loop

| Flag | Description |
|---|---|
| `--sim` | Simulation mode (start daemon with `--sim` first) |
| `--device cpu\|cuda\|mps` | Device for emotion models (default: cpu) |
| `--media-backend` | Reachy media backend (`default`, `gstreamer`, `webrtc`) |
| `--use-webcam N` | Use webcam N instead of Reachy camera |
| `--no-audio` | Disable microphone input |
| `--no-announce` | Disable TTS emotion announcement via speaker |
| `--no-gaze` | Disable face gaze tracking |
| `--no-robot` | No Reachy: webcam + logging only |

### Options — conversation

| Flag | Description |
|---|---|
| `--text` | Type input instead of speaking |
| `--lang CODE` | STT/TTS language, e.g. `en-US`, `fr-FR` (default: `en-US`) |
| `--model NAME` | Gemini model (default: `gemini-2.0-flash`) |
| `--prompt TEXT` | Override the system prompt |
| `--sim` | Simulation mode |
| `--media-backend` | Reachy media backend |

## Project Structure

```
reachy-emotion/
├── .env.example                      # API key template (copy to .env)
├── .gitignore
├── index.html                        # HF Space landing page
├── style.css
├── pyproject.toml                    # Package config + all dependencies
├── README.md
└── src/reachy_emotion/
    ├── __init__.py
    ├── main.py                       # ReachyEmotionApp + CLI (emotion loop)
    ├── reachy_handler.py             # ActionCommand → Motion, RecordedMoves, TTS
    ├── tts_announcer.py              # speak_text() + announce_emotion() via gTTS/pydub
    ├── gemini_bridge.py              # Gemini chat session + detect_emotion tool
    ├── voice_input.py                # STT from Reachy mic via SpeechRecognition
    └── conversation_app.py           # ReachyConversationApp + CLI (Gemini chat)
```
