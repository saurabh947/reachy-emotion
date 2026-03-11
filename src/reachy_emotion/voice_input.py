"""Voice input: capture speech from Reachy's mic and transcribe via Google Speech API.

Requirements:
    pip install SpeechRecognition
    Internet access for Google Speech Recognition (free quota).

The listen() function records audio from Reachy's microphone array until it detects
silence after speech (energy-based VAD), then transcribes it.
"""

import io
import logging
import time
import wave
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Energy-based VAD thresholds (normalised float32 RMS values)
_SPEECH_ENERGY_THRESHOLD = 0.015   # RMS above this → speech detected
_SILENCE_AFTER_SPEECH_SEC = 1.2    # stop recording after this much silence
_MIN_SPEECH_SEC = 0.3              # discard recordings shorter than this
_MAX_RECORDING_SEC = 12.0          # hard cap per utterance
_POLL_SLEEP_SEC = 0.02             # sleep between get_audio_sample() polls


def _numpy_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert a mono float32 numpy array to in-memory WAV bytes (PCM 16-bit)."""
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def listen(
    mini: Any,
    sample_rate: int = 16000,
    language: str = "en-US",
) -> str | None:
    """Record speech from Reachy's mic and return the transcribed text.

    Uses simple energy-based VAD: starts recording when RMS exceeds
    _SPEECH_ENERGY_THRESHOLD, stops when silence resumes for
    _SILENCE_AFTER_SPEECH_SEC. Falls back to None on any failure.

    Args:
        mini: ReachyMini instance with media.get_audio_sample().
        sample_rate: Reachy's microphone sample rate (default 16 kHz).
        language: BCP-47 language code for Google Speech Recognition.

    Returns:
        Transcribed text string, or None if nothing was understood.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        logger.error("SpeechRecognition not installed. Run: pip install SpeechRecognition")
        return None

    logger.debug("Listening… (speak now)")

    chunks: list[np.ndarray] = []
    speech_started = False
    silence_start: float | None = None
    record_start = time.time()

    while True:
        now = time.time()
        elapsed = now - record_start

        if elapsed > _MAX_RECORDING_SEC:
            logger.debug("Max recording time reached")
            break

        chunk = mini.media.get_audio_sample()
        if chunk is None:
            time.sleep(_POLL_SLEEP_SEC)
            continue

        # Stereo → mono
        if chunk.ndim > 1:
            chunk = chunk.mean(axis=1)
        chunk = chunk.astype(np.float32)

        energy = float(np.sqrt(np.mean(chunk ** 2)))

        if energy > _SPEECH_ENERGY_THRESHOLD:
            speech_started = True
            silence_start = None
            chunks.append(chunk)
        elif speech_started:
            # Speech has started; track silence duration
            chunks.append(chunk)
            if silence_start is None:
                silence_start = now
            elif now - silence_start >= _SILENCE_AFTER_SPEECH_SEC:
                logger.debug("Silence detected — stopping")
                break
        # else: still waiting for speech to start

    if not chunks:
        return None

    audio_arr = np.concatenate(chunks)
    duration = len(audio_arr) / sample_rate

    if duration < _MIN_SPEECH_SEC:
        logger.debug("Recording too short (%.2fs) — discarding", duration)
        return None

    wav_bytes = _numpy_to_wav_bytes(audio_arr, sample_rate)
    audio_data = sr.AudioData(wav_bytes, sample_rate, sample_width=2)

    recognizer = sr.Recognizer()
    try:
        text: str = recognizer.recognize_google(audio_data, language=language)
        logger.info("Heard: %s", text)
        return text
    except sr.UnknownValueError:
        logger.debug("Speech not understood")
        return None
    except sr.RequestError as exc:
        logger.warning("Google STT request failed: %s", exc)
        return None
