"""Tests for voice_input: _numpy_to_wav_bytes() and listen()."""

import io
import sys
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _numpy_to_wav_bytes
# ---------------------------------------------------------------------------

def test_wav_bytes_produces_valid_wav_header():
    from reachy_emotion.voice_input import _numpy_to_wav_bytes

    audio = np.zeros(16000, dtype=np.float32)  # 1 s silence
    wav_bytes = _numpy_to_wav_bytes(audio, sample_rate=16000)

    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() == 16000


def test_wav_bytes_clips_positive_overflow():
    from reachy_emotion.voice_input import _numpy_to_wav_bytes

    audio = np.array([2.0], dtype=np.float32)
    wav_bytes = _numpy_to_wav_bytes(audio, sample_rate=16000)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(1)
    pcm = np.frombuffer(raw, dtype=np.int16)
    assert pcm[0] == 32767


def test_wav_bytes_clips_negative_overflow():
    from reachy_emotion.voice_input import _numpy_to_wav_bytes

    audio = np.array([-2.0], dtype=np.float32)
    wav_bytes = _numpy_to_wav_bytes(audio, sample_rate=16000)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(1)
    pcm = np.frombuffer(raw, dtype=np.int16)
    assert pcm[0] == -32768


def test_wav_bytes_preserves_normal_amplitude():
    from reachy_emotion.voice_input import _numpy_to_wav_bytes

    audio = np.array([0.5], dtype=np.float32)
    wav_bytes = _numpy_to_wav_bytes(audio, sample_rate=16000)
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        raw = wf.readframes(1)
    pcm = np.frombuffer(raw, dtype=np.int16)
    # 0.5 * 32767 ≈ 16383
    assert abs(pcm[0] - 16383) <= 1


# ---------------------------------------------------------------------------
# listen — guard conditions
# ---------------------------------------------------------------------------

def test_listen_returns_none_when_speech_recognition_missing(monkeypatch):
    """If SpeechRecognition is not installed, listen() returns None gracefully."""
    monkeypatch.setitem(sys.modules, "speech_recognition", None)

    from reachy_emotion import voice_input
    # Reload to pick up the monkeypatched sys.modules for the import inside listen()
    mini = MagicMock()
    mini.media.get_audio_sample.return_value = None

    result = voice_input.listen(mini)
    assert result is None


def test_listen_returns_none_when_no_audio_ever_returned():
    """If mini.media.get_audio_sample always returns None, listen() returns None."""
    from reachy_emotion.voice_input import listen

    mini = MagicMock()
    mini.media.get_audio_sample.return_value = None

    with patch("reachy_emotion.voice_input._MAX_RECORDING_SEC", 0.05):
        result = listen(mini)

    assert result is None


def test_listen_returns_none_for_short_silent_audio():
    """Audio below speech energy threshold should produce None (discarded as too short)."""
    from reachy_emotion.voice_input import listen

    mini = MagicMock()
    # Silence: very low energy, well below _SPEECH_ENERGY_THRESHOLD
    silent_chunk = np.zeros(1600, dtype=np.float32)
    mini.media.get_audio_sample.return_value = silent_chunk

    with patch("reachy_emotion.voice_input._MAX_RECORDING_SEC", 0.05):
        result = listen(mini)

    assert result is None


def test_listen_converts_stereo_to_mono():
    """Stereo audio chunks must be converted to mono (mean of channels)."""
    from reachy_emotion import voice_input

    mini = MagicMock()
    # Stereo: left channel = 1.0, right = 0.0 → mono mean = 0.5
    stereo = np.zeros((1600, 2), dtype=np.float32)
    stereo[:, 0] = 1.0
    mini.media.get_audio_sample.return_value = stereo

    rms_values: list[float] = []
    original_sqrt = np.sqrt

    def spy_sqrt(x):
        val = original_sqrt(x)
        if isinstance(x, (float, np.floating)):
            rms_values.append(float(val))
        return val

    with patch("reachy_emotion.voice_input._MAX_RECORDING_SEC", 0.05), \
         patch("numpy.sqrt", side_effect=spy_sqrt):
        voice_input.listen(mini)

    # mono mean of (1.0, 0.0) = 0.5; RMS of all-0.5 = 0.5
    assert len(rms_values) > 0
    assert pytest.approx(rms_values[0], abs=0.01) == 0.5


# ---------------------------------------------------------------------------
# listen — STT error handling
# ---------------------------------------------------------------------------

def test_listen_returns_none_on_unknown_value_error():
    """Google STT returning UnknownValueError should yield None, not an exception."""
    import speech_recognition as sr
    from reachy_emotion.voice_input import listen

    mini = MagicMock()
    # A chunk with high energy so speech is "detected"
    loud_chunk = np.ones(1600, dtype=np.float32)
    mini.media.get_audio_sample.return_value = loud_chunk

    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.side_effect = sr.UnknownValueError()

    with patch("reachy_emotion.voice_input._MAX_RECORDING_SEC", 0.1), \
         patch("reachy_emotion.voice_input._SILENCE_AFTER_SPEECH_SEC", 0.05), \
         patch("speech_recognition.Recognizer", return_value=mock_recognizer), \
         patch("speech_recognition.AudioData"):
        result = listen(mini)

    assert result is None


def test_listen_returns_none_on_request_error():
    """Google STT RequestError should yield None, not an exception."""
    import speech_recognition as sr
    from reachy_emotion.voice_input import listen

    mini = MagicMock()
    loud_chunk = np.ones(1600, dtype=np.float32)
    mini.media.get_audio_sample.return_value = loud_chunk

    mock_recognizer = MagicMock()
    mock_recognizer.recognize_google.side_effect = sr.RequestError("network down")

    with patch("reachy_emotion.voice_input._MAX_RECORDING_SEC", 0.1), \
         patch("reachy_emotion.voice_input._SILENCE_AFTER_SPEECH_SEC", 0.05), \
         patch("speech_recognition.Recognizer", return_value=mock_recognizer), \
         patch("speech_recognition.AudioData"):
        result = listen(mini)

    assert result is None
