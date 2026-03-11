"""Tests for GeminiBridge: text extraction, function-call handling, tool execution."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal stubs for Gemini response objects
# ---------------------------------------------------------------------------

class _Part:
    def __init__(self, text=None, function_call=None):
        self.text = text
        self.function_call = function_call


class _Candidate:
    def __init__(self, parts):
        self.content = MagicMock(parts=parts)


class _Response:
    def __init__(self, *parts):
        self.candidates = [_Candidate(list(parts))]


def _fn_call(name: str) -> MagicMock:
    fc = MagicMock()
    fc.name = name
    return fc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bridge(mini=None):
    from reachy_emotion.gemini_bridge import GeminiBridge
    return GeminiBridge(api_key="test-key", mini=mini or MagicMock(), model="gemini-test")


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

def test_extract_text_single_part():
    bridge = _make_bridge()
    response = _Response(_Part(text="Hello there"))
    assert bridge._extract_text(response) == "Hello there"


def test_extract_text_multiple_parts_concatenated():
    bridge = _make_bridge()
    response = _Response(_Part(text="Hello "), _Part(text="world"))
    result = bridge._extract_text(response)
    assert "Hello" in result and "world" in result


def test_extract_text_ignores_parts_without_text():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc), _Part(text="Final reply"))
    assert bridge._extract_text(response) == "Final reply"


def test_extract_text_returns_empty_string_when_no_text():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc))
    assert bridge._extract_text(response) == ""


# ---------------------------------------------------------------------------
# _extract_function_calls
# ---------------------------------------------------------------------------

def test_extract_function_calls_empty_when_no_fn_call():
    bridge = _make_bridge()
    response = _Response(_Part(text="Just text"))
    assert bridge._extract_function_calls(response) == []


def test_extract_function_calls_finds_single_call():
    bridge = _make_bridge()
    fc = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc))
    calls = bridge._extract_function_calls(response)
    assert len(calls) == 1
    assert calls[0].name == "detect_emotion"


def test_extract_function_calls_finds_multiple_calls():
    bridge = _make_bridge()
    fc1 = _fn_call("detect_emotion")
    fc2 = _fn_call("detect_emotion")
    response = _Response(_Part(function_call=fc1), _Part(function_call=fc2))
    calls = bridge._extract_function_calls(response)
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# _run_emotion_detection
# ---------------------------------------------------------------------------

def test_run_emotion_detection_returns_error_when_no_detector():
    bridge = _make_bridge()
    bridge._emotion_detector = None
    result = bridge._run_emotion_detection()
    assert "error" in result


def test_run_emotion_detection_returns_unknown_when_no_frame():
    bridge = _make_bridge()
    bridge._mini = MagicMock()
    bridge._mini.media.get_frame.return_value = None
    bridge._emotion_detector = MagicMock()
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "unknown"
    assert result["faces_detected"] == 0


def test_run_emotion_detection_returns_neutral_when_process_frame_returns_none():
    bridge = _make_bridge()
    bridge._mini = MagicMock()
    bridge._mini.media.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    bridge._mini.media.get_audio_sample.return_value = None
    bridge._emotion_detector = MagicMock()
    bridge._emotion_detector.process_frame.return_value = None
    result = bridge._run_emotion_detection()
    assert result["dominant_emotion"] == "neutral"
    assert result["confidence"] == 0.0


def test_run_emotion_detection_returns_correct_emotion_dict():
    bridge = _make_bridge()
    bridge._mini = MagicMock()
    bridge._mini.media.get_frame.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    bridge._mini.media.get_audio_sample.return_value = None

    mock_result = MagicMock()
    mock_result.emotion.dominant_emotion.value = "happy"
    mock_result.emotion.confidence = 0.93
    mock_result.detection.faces = [MagicMock()]
    mock_result.action.action_type = "acknowledge"

    bridge._emotion_detector = MagicMock()
    bridge._emotion_detector.process_frame.return_value = mock_result

    result = bridge._run_emotion_detection()

    assert result["dominant_emotion"] == "happy"
    assert result["confidence"] == pytest.approx(0.93, abs=0.01)
    assert result["faces_detected"] == 1
    assert result["action_suggested"] == "acknowledge"
    assert bridge._last_emotion_result is mock_result


def test_run_emotion_detection_converts_stereo_audio_to_mono():
    """Stereo audio from Reachy's mic must be converted to mono before detection."""
    bridge = _make_bridge()
    bridge._mini = MagicMock()
    bridge._mini.media.get_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

    stereo = np.ones((1600, 2), dtype=np.float32)
    bridge._mini.media.get_audio_sample.return_value = stereo

    bridge._emotion_detector = MagicMock()
    bridge._emotion_detector.process_frame.return_value = None

    bridge._run_emotion_detection()

    call_args = bridge._emotion_detector.process_frame.call_args
    audio_arg = call_args.kwargs.get("audio") or call_args[1].get("audio")
    assert audio_arg is not None
    assert audio_arg.ndim == 1  # mono


# ---------------------------------------------------------------------------
# chat — top-level orchestration
# ---------------------------------------------------------------------------

def test_chat_raises_if_not_initialized():
    bridge = _make_bridge()
    with pytest.raises(RuntimeError, match="initialize"):
        bridge.chat("Hello!")


def test_chat_returns_text_when_no_tool_calls():
    bridge = _make_bridge()
    bridge._chat = MagicMock()
    bridge._chat.send_message.return_value = _Response(_Part(text="Nice to meet you!"))

    text, emotion = bridge.chat("Hello Reachy!")

    assert text == "Nice to meet you!"
    assert emotion is None
    bridge._chat.send_message.assert_called_once_with("Hello Reachy!")


def test_chat_resets_last_emotion_result_each_turn():
    bridge = _make_bridge()
    bridge._chat = MagicMock()
    bridge._last_emotion_result = MagicMock()  # leftover from previous turn
    bridge._chat.send_message.return_value = _Response(_Part(text="Hi"))

    _, emotion = bridge.chat("Hello")

    assert emotion is None  # reset at start of turn


def test_chat_handles_detect_emotion_tool_call():
    """Full tool-call loop: Gemini calls detect_emotion, gets result, returns text."""
    from google.genai import types as genai_types

    bridge = _make_bridge()
    bridge._chat = MagicMock()

    # Turn 1: Gemini returns a function call
    fc = _fn_call("detect_emotion")
    tool_response = _Response(_Part(function_call=fc))
    # Turn 2: Gemini returns text after receiving the tool result
    final_response = _Response(_Part(text="You look happy!"))

    bridge._chat.send_message.side_effect = [tool_response, final_response]

    # Mock emotion detection
    mock_er = MagicMock()
    mock_er.emotion.dominant_emotion.value = "happy"
    mock_er.emotion.confidence = 0.88
    mock_er.detection.faces = [MagicMock()]
    mock_er.action.action_type = "acknowledge"

    bridge._mini = MagicMock()
    bridge._mini.media.get_frame.return_value = np.zeros((100, 100, 3), dtype=np.uint8)
    bridge._mini.media.get_audio_sample.return_value = None
    bridge._emotion_detector = MagicMock()
    bridge._emotion_detector.process_frame.return_value = mock_er

    mock_part = MagicMock()
    with patch.object(genai_types.Part, "from_function_response", return_value=mock_part):
        text, emotion = bridge.chat("How am I feeling?")

    assert text == "You look happy!"
    assert emotion is mock_er
    assert bridge._chat.send_message.call_count == 2


def test_chat_handles_unknown_tool_gracefully():
    """If Gemini calls an unknown tool, the bridge sends an error response and continues."""
    from google.genai import types as genai_types

    bridge = _make_bridge()
    bridge._chat = MagicMock()

    fc = _fn_call("nonexistent_tool")
    tool_response = _Response(_Part(function_call=fc))
    final_response = _Response(_Part(text="Okay."))

    bridge._chat.send_message.side_effect = [tool_response, final_response]
    bridge._emotion_detector = MagicMock()

    mock_part = MagicMock()
    with patch.object(genai_types.Part, "from_function_response", return_value=mock_part):
        text, emotion = bridge.chat("Do something weird")

    assert text == "Okay."
    assert emotion is None
