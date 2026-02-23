"""TTS Announcer: converts emotion to speech and plays via ReachyMini SDK Speaker."""

import os
import tempfile
from typing import TYPE_CHECKING

from gtts import gTTS
from pydub import AudioSegment

if TYPE_CHECKING:
    from reachy_mini import ReachyMini


def announce_emotion(emotion: str, mini: "ReachyMini", lang: str = "en") -> bool:
    """Generate speech for detected emotion and play through Reachy's speaker.

    Args:
        emotion: Emotion label (e.g. happy, sad, angry).
        mini: ReachyMini instance (SDK provides play_sound).
        lang: Language for TTS (default: en).

    Returns:
        True if announcement succeeded.
    """
    if not emotion or not mini:
        return False
    try:
        text = f"I detect you seem {emotion}"
        tts = gTTS(text=text, lang=lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
            tts.save(mp3_f.name)
            try:
                # Reachy Mini expects WAV; convert MP3 to WAV
                audio = AudioSegment.from_mp3(mp3_f.name)
                audio = audio.set_frame_rate(16000).set_channels(1)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
                    audio.export(wav_f.name, format="wav")
                    try:
                        mini.media.play_sound(wav_f.name)
                        return True
                    finally:
                        try:
                            os.unlink(wav_f.name)
                        except OSError:
                            pass
            finally:
                try:
                    os.unlink(mp3_f.name)
                except OSError:
                    pass
    except Exception:
        return False
