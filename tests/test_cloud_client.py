"""Tests for EmotionCloudClient: on-demand detect_emotion, lifecycle, stub loading."""

import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mini(frame=None):
    mini = MagicMock()
    mini.media.get_frame.return_value = (
        frame if frame is not None
        else np.zeros((480, 640, 3), dtype=np.uint8)
    )
    return mini


def _make_client(mini=None, endpoint="1.2.3.4:50051"):
    from reachy_emotion.cloud_client import EmotionCloudClient
    return EmotionCloudClient(mini=mini or _make_mini(), endpoint=endpoint)


def _mock_response(dominant_emotion="happy", confidence=0.9,
                   stress=0.1, engagement=0.8, arousal=0.6,
                   buffering=False, error=""):
    r = MagicMock()
    r.dominant_emotion = dominant_emotion
    r.overall_confidence = confidence
    r.confidence_scores = {dominant_emotion: confidence}
    r.stress = stress
    r.engagement = engagement
    r.arousal = arousal
    r.buffering = buffering
    r.error = error
    return r


# ---------------------------------------------------------------------------
# detect_emotion — before start()
# ---------------------------------------------------------------------------

def test_detect_emotion_returns_unclear_before_start():
    """Calling detect_emotion() without start() should return 'unclear', not raise."""
    client = _make_client()
    result = client.detect_emotion()
    assert result["dominant_emotion"] == "unclear"
    assert "note" in result


# ---------------------------------------------------------------------------
# detect_emotion — gRPC interaction
# ---------------------------------------------------------------------------

def test_detect_emotion_returns_result_on_success():
    """On a successful stream response, detect_emotion returns the emotion dict."""
    client = _make_client()
    client._pb2 = MagicMock()
    client._pb2_grpc = MagicMock()

    response = _mock_response(dominant_emotion="happy", confidence=0.9)

    with patch("reachy_emotion.cloud_client.grpc") as mock_grpc:
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel
        stub = MagicMock()
        client._pb2_grpc.EmotionDetectionStub.return_value = stub
        stub.StreamEmotion.return_value = iter([response])

        result = client.detect_emotion()

    assert result["dominant_emotion"] == "happy"
    assert result["confidence"] == pytest.approx(0.9, abs=0.01)
    assert result["stress"] == pytest.approx(0.1, abs=0.01)


def test_detect_emotion_skips_buffering_responses():
    """detect_emotion must skip buffering=True responses and wait for a real one."""
    client = _make_client()
    client._pb2 = MagicMock()
    client._pb2_grpc = MagicMock()

    buffering = _mock_response(buffering=True)
    real = _mock_response(dominant_emotion="sad", confidence=0.75, buffering=False)

    with patch("reachy_emotion.cloud_client.grpc") as mock_grpc:
        mock_grpc.insecure_channel.return_value = MagicMock()
        stub = MagicMock()
        client._pb2_grpc.EmotionDetectionStub.return_value = stub
        stub.StreamEmotion.return_value = iter([buffering, buffering, real])

        result = client.detect_emotion()

    assert result["dominant_emotion"] == "sad"


def test_detect_emotion_skips_error_responses():
    """detect_emotion must skip responses with error set and continue."""
    client = _make_client()
    client._pb2 = MagicMock()
    client._pb2_grpc = MagicMock()

    bad = _mock_response(error="Backend error 413")
    good = _mock_response(dominant_emotion="neutral", confidence=0.6, error="")

    with patch("reachy_emotion.cloud_client.grpc") as mock_grpc:
        mock_grpc.insecure_channel.return_value = MagicMock()
        stub = MagicMock()
        client._pb2_grpc.EmotionDetectionStub.return_value = stub
        stub.StreamEmotion.return_value = iter([bad, good])

        result = client.detect_emotion()

    assert result["dominant_emotion"] == "neutral"


def test_detect_emotion_returns_unclear_on_grpc_error():
    """If the gRPC call raises, detect_emotion returns 'unclear' instead of propagating."""
    client = _make_client()
    client._pb2 = MagicMock()
    client._pb2_grpc = MagicMock()

    with patch("reachy_emotion.cloud_client.grpc") as mock_grpc:
        mock_grpc.insecure_channel.return_value = MagicMock()
        stub = MagicMock()
        client._pb2_grpc.EmotionDetectionStub.return_value = stub
        stub.StreamEmotion.side_effect = Exception("connection refused")

        result = client.detect_emotion()

    assert result["dominant_emotion"] == "unclear"
    assert "note" in result


def test_detect_emotion_closes_channel_on_success():
    """gRPC channel must be closed after a successful detect_emotion call."""
    client = _make_client()
    client._pb2 = MagicMock()
    client._pb2_grpc = MagicMock()

    response = _mock_response()

    with patch("reachy_emotion.cloud_client.grpc") as mock_grpc:
        mock_channel = MagicMock()
        mock_grpc.insecure_channel.return_value = mock_channel
        stub = MagicMock()
        client._pb2_grpc.EmotionDetectionStub.return_value = stub
        stub.StreamEmotion.return_value = iter([response])

        client.detect_emotion()

    mock_channel.close.assert_called_once()


# ---------------------------------------------------------------------------
# stop() — lifecycle
# ---------------------------------------------------------------------------

def test_stop_is_idempotent():
    """Calling stop() multiple times must not raise."""
    client = _make_client()
    client.stop()
    client.stop()


# ---------------------------------------------------------------------------
# _load_stubs — stub generation guard
# ---------------------------------------------------------------------------

def test_load_stubs_does_not_run_protoc_if_stubs_exist():
    """If emotion_pb2.py already exists on disk, protoc must not be invoked."""
    from pathlib import Path
    import reachy_emotion.cloud_client as cc_module

    proto_dir = Path(cc_module.__file__).parent / "proto"
    pb2_path = proto_dir / "emotion_pb2.py"

    if not pb2_path.exists():
        pytest.skip("Stubs not yet generated — run the app once to generate them")

    with patch("reachy_emotion.cloud_client.subprocess.run") as mock_run:
        with patch("reachy_emotion.cloud_client.importlib.util.spec_from_file_location") as mock_spec:
            spec = MagicMock()
            spec.loader.exec_module = MagicMock()
            mock_spec.return_value = spec
            try:
                cc_module._load_stubs()
            except Exception:
                pass  # import errors OK — we only verify protoc wasn't called

    mock_run.assert_not_called()


# ---------------------------------------------------------------------------
# session_id
# ---------------------------------------------------------------------------

def test_default_session_id_is_uuid():
    """Default session_id must be a valid UUID4 string."""
    client = _make_client()
    parsed = uuid.UUID(client._session_id)
    assert str(parsed) == client._session_id


def test_custom_session_id_is_used():
    from reachy_emotion.cloud_client import EmotionCloudClient
    client = EmotionCloudClient(mini=_make_mini(), endpoint="1.2.3.4:50051", session_id="robot-001")
    assert client._session_id == "robot-001"
