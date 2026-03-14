"""TTS Announcer: converts text to speech and plays via ReachyMini SDK Speaker.

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
from typing import TYPE_CHECKING

from gtts import gTTS

if TYPE_CHECKING:
    from reachy_mini import ReachyMini

logger = logging.getLogger(__name__)

# Thread-safe one-time warning for missing ffmpeg.
_ffmpeg_warning_lock = threading.Lock()
_FFMPEG_WARNING_SHOWN = False


def _ffmpeg_available() -> bool:
    """Check whether ffmpeg is on PATH (required by pydub for MP3 decoding)."""
    return shutil.which("ffmpeg") is not None


def speak_text(text: str, mini: "ReachyMini", lang: str = "en") -> bool:
    """Convert arbitrary text to speech and play through Reachy's speaker.

    Requires ffmpeg (pydub uses it for MP3 → WAV conversion).
    If ffmpeg is missing, logs a one-time error and returns False without crashing.

    Args:
        text: Text to speak aloud.
        mini: ReachyMini instance.
        lang: BCP-47 language code (default: "en").

    Returns:
        True if playback succeeded.
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

        # Create MP3 temp file — save *and* clean up inside one try/finally
        mp3_fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
        os.close(mp3_fd)
        try:
            tts.save(mp3_path)
            audio = AudioSegment.from_mp3(mp3_path)
        finally:
            try:
                os.unlink(mp3_path)
            except OSError:
                pass

        audio = audio.set_frame_rate(16000).set_channels(1)

        # Create WAV temp file — export and play inside one try/finally
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)
        try:
            audio.export(wav_path, format="wav")
            mini.media.play_sound(wav_path)
            return True
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    except Exception as e:
        logger.warning("TTS failed: %s", e)
        return False


def announce_emotion(emotion: str, mini: "ReachyMini", lang: str = "en") -> bool:
    """Generate speech for a detected emotion and play through Reachy's speaker.

    Args:
        emotion: Emotion label (e.g. happy, sad, angry).
        mini: ReachyMini instance.
        lang: Language for TTS (default: "en").

    Returns:
        True if announcement succeeded.
    """
    if not emotion:
        return False
    return speak_text(f"I detect you seem {emotion}", mini, lang=lang)
