# Reachy Emotion

Lightweight Reachy Mini app that detects human emotions and responds with robot actions and speech.

## Architecture

- **Emotion Detection SDK**: Uses camera + microphone to detect emotions, outputs `ActionCommand`
- **Reachy Mini Domain**: Receives `ActionCommand`, invokes ReachyMini SDK (Motion, RecordedMoves, Speaker)

## Prerequisites

1. **Reachy Mini daemon** (for real robot or simulation):
   ```bash
   uv run reachy-mini-daemon --sim   # Simulation
   uv run reachy-mini-daemon         # Real robot
   ```

2. **Emotion Detection SDK** (install from local path first):
   ```bash
   pip install -e ~/Documents/Study/Robotics/MyApps/emotion-detection-action
   ```

## Installation

1. Install emotion-detection-action first (local path):
   ```bash
   pip install -e ~/Documents/Study/Robotics/MyApps/emotion-detection-action
   ```

2. Install reachy-emotion:
   ```bash
   cd reachy-emotion
   pip install -e .
   ```

## Usage

**With Reachy Mini** (camera + mic from robot):
```bash
python -m reachy_emotion.main
```

**With webcam** (no Reachy hardware, for testing):
```bash
python -m reachy_emotion.main --no-robot --use-webcam 0
```

**Options**:
- `--sim` - Use Reachy daemon simulation (run daemon with `--sim` first)
- `--device cpu|cuda|mps` - Device for emotion models (default: cpu)
- `--media-backend default|gstreamer|webrtc` - Reachy media backend
- `--use-webcam N` - Use webcam N instead of Reachy camera
- `--no-audio` - Disable microphone
- `--no-announce` - Disable TTS emotion announcement
- `--no-robot` - No Reachy: webcam + logging only

## Project Structure

```
reachy-emotion/
├── src/reachy_emotion/
│   ├── reachy_handler.py   # ActionCommand → SDK Motion, RecordedMoves, TTS
│   ├── tts_announcer.py    # TTS → SDK Speaker
│   └── main.py             # Entry point
└── pyproject.toml
```
