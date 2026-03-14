"""TTS Announcer: converts text to speech and plays via ReachyMini SDK speaker.

Uses push_audio_sample() — the official Reachy Mini SDK audio output API.
Call mini.media.start_playing() once before the conversation loop and
mini.media.stop_playing() once after it ends; speak_text() assumes the
speaker is already activated.

System requirement: pydub uses ffmpeg to decode MP3.
Install ffmpeg before running:
    macOS:  brew install ffmpeg
    Ubuntu: sudo apt install ffmpeg
    Windows: https://ffmpeg.org/download.html
"""

import logging
import os
import shutil
import tempfile
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
from gtts import gTTS

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# Thread-safe one-time warning for missing ffmpeg.
_ffmpeg_warning_lock = threading.Lock()
_FFMPEG_WARNING_SHOWN = False

# Reachy Mini speaker sample rate (16 kHz per SDK docs).
_SPEAKER_SAMPLE_RATE = 16000


def _ffmpeg_available() -> bool:
    """Check whether ffmpeg is on PATH (required by pydub for MP3 decoding)."""
    return shutil.which("ffmpeg") is not None


def speak_text(text: str, mini: "ReachyMini", lang: str = "en") -> bool:
    """Convert arbitrary text to speech and play through Reachy's speaker.

    Uses push_audio_sample() per the official SDK audio API.
    Assumes mini.media.start_playing() has already been called by the caller.
    Requires ffmpeg (pydub uses it for MP3 → WAV conversion).
    If ffmpeg is missing, logs a one-time error and returns False without crashing.

    Args:
        text: Text to speak aloud.
        mini: Connected ReachyMini instance (speaker must already be started).
        lang: BCP-47 language code (default: "en").

    Returns:
        True if playback was initiated successfully.
    """
    global _FFMPEG_WARNING_SHOWN

    if not text or not mini:
        return False

    if not _ffmpeg_available():
        with _ffmpeg_warning_lock:
            if not _FFMPEG_WARNING_SHOWN:
                logger.error(
                    "ffmpeg not found — TTS disabled. "
                    "Install: brew install ffmpeg (macOS) | sudo apt install ffmpeg (Ubuntu)"
                )
                _FFMPEG_WARNING_SHOWN = True
        return False

    try:
        from pydub import AudioSegment

        tts = gTTS(text=text, lang=lang, slow=False)

        # Generate MP3, convert to 16 kHz mono, load as float32 numpy array
        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(mp3_fd)
        try:
            tts.save(mp3_path)
            audio_seg = AudioSegment.from_mp3(mp3_path)
        finally:
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

        audio_seg = audio_seg.set_frame_rate(_SPEAKER_SAMPLE_RATE).set_channels(1)
        raw_bytes = audio_seg.raw_data
        # pydub gives int16 PCM; SDK expects float32 in range [-1, 1], shape (N, channels)
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        samples = samples.reshape(-1, 1)

        duration_sec = len(samples) / _SPEAKER_SAMPLE_RATE

        mini.media.push_audio_sample(samples)
        # push_audio_sample is non-blocking; sleep so audio finishes before returning
        time.sleep(duration_sec)
        return True

    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return False


def announce_emotion(emotion: str, mini: "ReachyMini", lang: str = "en") -> bool:
    """Generate speech for a detected emotion and play through Reachy's speaker.

    Args:
        emotion: Emotion label (e.g. happy, sad, angry).
        mini: ReachyMini instance (speaker must already be started).
        lang: Language for TTS (default: "en").

    Returns:
        True if announcement was initiated successfully.
    """
    if not emotion:
        return False
    return speak_text(f"I detect you seem {emotion}", mini, lang=lang)
