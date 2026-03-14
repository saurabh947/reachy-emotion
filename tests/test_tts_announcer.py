"""Tests for tts_announcer: speak_text() and announce_emotion()."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# speak_text — guard conditions
# ---------------------------------------------------------------------------

def test_speak_text_returns_false_for_empty_text():
    from reachy_emotion.tts_announcer import speak_text

    assert speak_text("", MagicMock()) is False


def test_speak_text_returns_false_for_none_mini():
    from reachy_emotion.tts_announcer import speak_text

    assert speak_text("hello", None) is False


# ---------------------------------------------------------------------------
# speak_text — ffmpeg guard
# ---------------------------------------------------------------------------

def test_speak_text_returns_false_when_ffmpeg_missing():
    from reachy_emotion import tts_announcer
    tts_announcer._FFMPEG_WARNING_SHOWN = False

    with patch("reachy_emotion.tts_announcer._ffmpeg_available", return_value=False):
        result = tts_announcer.speak_text("hello", MagicMock())

    assert result is False


def test_speak_text_logs_ffmpeg_error_exactly_once():
    """The one-time ffmpeg error must not spam the log on repeated calls."""
    from reachy_emotion import tts_announcer
    tts_announcer._FFMPEG_WARNING_SHOWN = False

    mini = MagicMock()
    with patch("reachy_emotion.tts_announcer._ffmpeg_available", return_value=False):
        with patch.object(tts_announcer.logger, "error") as mock_error:
            tts_announcer.speak_text("first", mini)
            tts_announcer.speak_text("second", mini)
            tts_announcer.speak_text("third", mini)

    assert mock_error.call_count == 1


# ---------------------------------------------------------------------------
# speak_text — happy path (ffmpeg present, TTS succeeds)
# ---------------------------------------------------------------------------

def test_speak_text_returns_true_on_success():
    """Full happy-path: gTTS + pydub succeed → play_sound called → returns True."""
    from reachy_emotion import tts_announcer
    tts_announcer._FFMPEG_WARNING_SHOWN = False

    mini = MagicMock()
    mock_segment = MagicMock()
    mock_segment.set_frame_rate.return_value = mock_segment
    mock_segment.set_channels.return_value = mock_segment

    mock_gtts_instance = MagicMock()

    with patch("reachy_emotion.tts_announcer._ffmpeg_available", return_value=True), \
         patch("reachy_emotion.tts_announcer.gTTS", return_value=mock_gtts_instance), \
         patch("pydub.AudioSegment.from_mp3", return_value=mock_segment), \
         patch("reachy_emotion.tts_announcer.tempfile.mkstemp") as mock_mkstemp, \
         patch("reachy_emotion.tts_announcer.os.close"), \
         patch("reachy_emotion.tts_announcer.os.unlink"):

        # First mkstemp call → MP3 path, second → WAV path
        mock_mkstemp.side_effect = [(0, "/tmp/fake_tts.mp3"), (0, "/tmp/fake_tts.wav")]

        result = tts_announcer.speak_text("hello reachy", mini)

    assert result is True
    mini.media.play_sound.assert_called_once_with("/tmp/fake_tts.wav")


def test_speak_text_cleans_up_mp3_even_when_pydub_fails():
    """If AudioSegment.from_mp3 raises, the MP3 temp file must still be deleted."""
    from reachy_emotion import tts_announcer
    tts_announcer._FFMPEG_WARNING_SHOWN = False

    deleted_paths: list[str] = []

    def track_unlink(path: str) -> None:
        deleted_paths.append(path)

    with patch("reachy_emotion.tts_announcer._ffmpeg_available", return_value=True), \
         patch("reachy_emotion.tts_announcer.gTTS", return_value=MagicMock()), \
         patch("pydub.AudioSegment.from_mp3", side_effect=RuntimeError("decode fail")), \
         patch("reachy_emotion.tts_announcer.tempfile.mkstemp", return_value=(0, "/tmp/fake.mp3")), \
         patch("reachy_emotion.tts_announcer.os.close"), \
         patch("reachy_emotion.tts_announcer.os.unlink", side_effect=track_unlink):

        result = tts_announcer.speak_text("hello", MagicMock())

    assert result is False
    assert "/tmp/fake.mp3" in deleted_paths


def test_speak_text_returns_false_on_gtts_exception():
    from reachy_emotion import tts_announcer
    tts_announcer._FFMPEG_WARNING_SHOWN = False

    with patch("reachy_emotion.tts_announcer._ffmpeg_available", return_value=True), \
         patch("reachy_emotion.tts_announcer.gTTS", side_effect=RuntimeError("network error")), \
         patch("reachy_emotion.tts_announcer.tempfile.mkstemp", return_value=(0, "/tmp/fake.mp3")), \
         patch("reachy_emotion.tts_announcer.os.close"), \
         patch("reachy_emotion.tts_announcer.os.unlink"):

        result = tts_announcer.speak_text("hello", MagicMock())

    assert result is False


# ---------------------------------------------------------------------------
# announce_emotion
# ---------------------------------------------------------------------------

def test_announce_emotion_builds_correct_phrase():
    from reachy_emotion.tts_announcer import announce_emotion

    mini = MagicMock()
    with patch("reachy_emotion.tts_announcer.speak_text", return_value=True) as mock_speak:
        announce_emotion("happy", mini)

    mock_speak.assert_called_once_with("I detect you seem happy", mini, lang="en")


def test_announce_emotion_returns_false_for_empty_emotion():
    from reachy_emotion.tts_announcer import announce_emotion

    assert announce_emotion("", MagicMock()) is False


def test_announce_emotion_passes_lang():
    from reachy_emotion.tts_announcer import announce_emotion

    mini = MagicMock()
    with patch("reachy_emotion.tts_announcer.speak_text", return_value=True) as mock_speak:
        announce_emotion("triste", mini, lang="fr")

    mock_speak.assert_called_once_with("I detect you seem triste", mini, lang="fr")


# ---------------------------------------------------------------------------
# _ffmpeg_available
# ---------------------------------------------------------------------------

def test_ffmpeg_available_returns_true_when_on_path():
    from reachy_emotion.tts_announcer import _ffmpeg_available

    with patch("reachy_emotion.tts_announcer.shutil.which", return_value="/usr/bin/ffmpeg"):
        assert _ffmpeg_available() is True


def test_ffmpeg_available_returns_false_when_not_on_path():
    from reachy_emotion.tts_announcer import _ffmpeg_available

    with patch("reachy_emotion.tts_announcer.shutil.which", return_value=None):
        assert _ffmpeg_available() is False
