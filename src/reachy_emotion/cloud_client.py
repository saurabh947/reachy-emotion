"""EmotionCloudClient: on-demand gRPC emotion inference via emotion-cloud.

emotion-cloud runs on Google Cloud Kubernetes (GKE) and performs neural emotion
detection using a Two-Tower Multimodal Transformer (ViT-B/16 video + emotion2vec
audio).

This client contacts emotion-cloud **only when asked**.  When GeminiBridge calls
:meth:`detect_emotion`, the client captures camera frames, streams them until
the cloud returns a result, then closes the stream.

Camera strategy (bulletproof):
  1. Try ``mini.media.get_frame()`` — daemon IPC (preferred, zero-overhead).
  2. If that returns None, fall back to ``mini.release_media()`` + direct
     ``cv2.VideoCapture`` (the SDK-documented pattern for bypassing the daemon).
     After detection, ``mini.acquire_media()`` restores normal SDK operation.

Usage::

    client = EmotionCloudClient(mini=mini, endpoint="8.x.x.x:50051")
    client.start()

    result = client.detect_emotion()  # blocks ~2-3 s, returns dict
    print(result["dominant_emotion"], result["confidence"])

    client.stop()
"""

import importlib.util
import logging
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import grpc
import numpy as np

logger = logging.getLogger(__name__)

# Target frame rate while streaming during a detect_emotion call.
_STREAM_FPS = 15
_FRAME_INTERVAL = 1.0 / _STREAM_FPS

# Frames are resized to this before sending — matches ViT-B/16 input size and
# keeps each gRPC message ~150 KB instead of several MB.
_FRAME_SIZE = (224, 224)

# How many OpenCV device indices to probe when looking for a camera.
_MAX_CAMERA_INDEX = 5


# ---------------------------------------------------------------------------
# Proto stub loader
# ---------------------------------------------------------------------------

def _load_stubs() -> tuple[Any, Any]:
    """Return (emotion_pb2, emotion_pb2_grpc), generating stubs if needed."""
    proto_dir = Path(__file__).parent / "proto"
    pb2_path = proto_dir / "emotion_pb2.py"

    if not pb2_path.exists():
        logger.info("Generating gRPC stubs from proto/emotion.proto …")
        result = subprocess.run(
            [
                sys.executable, "-m", "grpc_tools.protoc",
                f"-I{proto_dir}",
                f"--python_out={proto_dir}",
                f"--grpc_python_out={proto_dir}",
                str(proto_dir / "emotion.proto"),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"grpc_tools.protoc failed (is grpcio-tools installed?):\n{result.stderr}"
            )
        logger.info("gRPC stubs generated at %s", proto_dir)

    proto_dir_str = str(proto_dir)
    if proto_dir_str not in sys.path:
        sys.path.insert(0, proto_dir_str)

    def _dynamic_import(module_name: str, file_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod

    pb2 = _dynamic_import("emotion_pb2", pb2_path)
    pb2_grpc = _dynamic_import("emotion_pb2_grpc", proto_dir / "emotion_pb2_grpc.py")
    return pb2, pb2_grpc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EmotionCloudClient:
    """On-demand emotion inference client for emotion-cloud.

    Call :meth:`detect_emotion` to capture frames, stream them to the cloud,
    and get the current emotional state.

    Args:
        mini: Connected ReachyMini instance.
        endpoint: emotion-cloud gRPC address, e.g. ``"8.x.x.x:50051"``.
        session_id: Stable identifier sent on every frame. Defaults to a
            fresh UUID so each conversation gets its own server-side buffer.
    """

    def __init__(
        self,
        mini: Any,
        endpoint: str,
        session_id: str | None = None,
    ) -> None:
        self._mini = mini
        self._endpoint = endpoint
        self._session_id = session_id or str(uuid.uuid4())
        self._pb2: Any = None
        self._pb2_grpc: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load proto stubs (runs protoc once if stubs are not yet on disk)."""
        self._pb2, self._pb2_grpc = _load_stubs()
        logger.info(
            "EmotionCloudClient ready (endpoint=%s, session=%s)",
            self._endpoint,
            self._session_id,
        )

    def stop(self) -> None:
        """Clean up (no-op — kept for API symmetry)."""
        logger.debug("EmotionCloudClient stopped")

    def health_check(self, timeout: float = 5.0) -> dict:
        """Perform a synchronous HealthCheck RPC against emotion-cloud.

        Returns:
            Dict with ``healthy``, ``model_status``, ``active_sessions``.

        Raises:
            RuntimeError: If :meth:`start` has not been called.
        """
        if self._pb2 is None or self._pb2_grpc is None:
            raise RuntimeError("EmotionCloudClient.start() must be called before health_check()")

        channel = grpc.insecure_channel(
            self._endpoint,
            options=[("grpc.max_receive_message_length", 1024 * 1024)],
        )
        try:
            stub = self._pb2_grpc.EmotionDetectionStub(channel)
            response = stub.HealthCheck(self._pb2.HealthRequest(), timeout=timeout)
            return {
                "healthy": response.healthy,
                "model_status": response.model_status,
                "active_sessions": response.active_sessions,
            }
        finally:
            channel.close()

    # ------------------------------------------------------------------
    # On-demand emotion detection
    # ------------------------------------------------------------------

    def detect_emotion(self, timeout: float = 15.0) -> dict:
        """Capture camera frames and return an emotion result from emotion-cloud.

        Camera fallback chain:
          1. ``mini.media.get_frame()`` — daemon IPC (fast, no overhead).
          2. ``mini.release_media()`` + ``cv2.VideoCapture`` — direct capture
             when the daemon's camera pipeline is unavailable (e.g. macOS
             permission granted to terminal but daemon didn't detect camera).
             Restores SDK media after capture via ``mini.acquire_media()``.

        Args:
            timeout: Maximum seconds to wait before giving up.

        Returns:
            Dict with keys: dominant_emotion, confidence, confidence_scores,
            stress, engagement, arousal. Returns an ``"unclear"`` fallback
            on camera or cloud errors.
        """
        if self._pb2 is None or self._pb2_grpc is None:
            return {"dominant_emotion": "unclear", "confidence": 0.0,
                    "note": "EmotionCloudClient.start() was not called"}

        # --- Determine camera source ---
        opencv_cap = None
        released_media = False

        # Probe: does the daemon's IPC camera work?
        sdk_ok = False
        try:
            if self._mini.media.get_frame() is not None:
                sdk_ok = True
        except Exception:
            pass

        if sdk_ok:
            logger.debug("detect_emotion: using daemon camera (IPC)")
            get_frame = self._mini.media.get_frame
        else:
            # Daemon camera not available — fall back to direct capture.
            # Per SDK docs: release_media() → cv2.VideoCapture → acquire_media()
            logger.info("detect_emotion: daemon camera unavailable, trying direct capture")
            try:
                self._mini.release_media()
                released_media = True
            except Exception as exc:
                logger.debug("release_media failed (continuing anyway): %s", exc)

            opencv_cap = self._open_camera()
            if opencv_cap is None:
                if released_media:
                    self._restore_media()
                return {"dominant_emotion": "unclear", "confidence": 0.0,
                        "note": "No camera available — grant camera permission to your terminal app "
                                "(System Settings → Privacy & Security → Camera) and restart the daemon"}

            def get_frame():
                ret, frame = opencv_cap.read()
                return frame if ret else None

        # --- Stream frames to emotion-cloud ---
        stop = threading.Event()

        def _request_gen():
            deadline = time.monotonic() + timeout
            while not stop.is_set() and time.monotonic() < deadline:
                frame_start = time.monotonic()
                try:
                    frame = get_frame()
                    if frame is not None:
                        # Camera returns BGR; emotion-cloud expects RGB.
                        if frame.ndim == 3 and frame.shape[2] == 3:
                            frame_rgb = frame[:, :, ::-1].copy()
                        else:
                            frame_rgb = frame
                        frame_rgb = cv2.resize(
                            frame_rgb, _FRAME_SIZE, interpolation=cv2.INTER_LINEAR,
                        )
                        h, w = frame_rgb.shape[:2]
                        yield self._pb2.EmotionRequest(
                            session_id=self._session_id,
                            video_frame=frame_rgb.astype(np.uint8).tobytes(),
                            frame_width=w,
                            frame_height=h,
                            timestamp_ms=int(time.time() * 1000),
                        )
                except Exception as exc:
                    logger.debug("Frame capture error (skipping): %s", exc)

                elapsed = time.monotonic() - frame_start
                time.sleep(max(0.0, _FRAME_INTERVAL - elapsed))

        channel = grpc.insecure_channel(
            self._endpoint,
            options=[
                ("grpc.max_send_message_length", 10 * 1024 * 1024),
                ("grpc.max_receive_message_length", 10 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
            ],
        )
        result = None
        try:
            stub = self._pb2_grpc.EmotionDetectionStub(channel)
            for response in stub.StreamEmotion(_request_gen()):
                if response.error:
                    logger.warning("emotion-cloud error: %s", response.error)
                    continue
                if response.buffering:
                    logger.debug("emotion-cloud: warming buffer…")
                    continue
                result = {
                    "dominant_emotion": response.dominant_emotion,
                    "confidence": round(response.overall_confidence, 2),
                    "confidence_scores": dict(response.confidence_scores),
                    "stress": round(response.stress, 2),
                    "engagement": round(response.engagement, 2),
                    "arousal": round(response.arousal, 2),
                }
                break
        except Exception as exc:
            logger.warning("emotion-cloud detect_emotion failed: %s", exc)
        finally:
            stop.set()
            if opencv_cap is not None:
                try:
                    opencv_cap.release()
                except Exception:
                    pass
            if released_media:
                self._restore_media()
            try:
                channel.close()
            except Exception:
                pass

        if result is None:
            return {
                "dominant_emotion": "unclear",
                "confidence": 0.0,
                "note": "No result from emotion-cloud within timeout",
            }
        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _open_camera(self) -> cv2.VideoCapture | None:
        """Scan device indices and return a working VideoCapture, or None."""
        for idx in range(_MAX_CAMERA_INDEX):
            try:
                cap = cv2.VideoCapture(idx)
                if not cap.isOpened():
                    cap.release()
                    continue
                ret, frame = cap.read()
                if ret and frame is not None:
                    logger.info("detect_emotion: opened camera at index %d", idx)
                    return cap
                cap.release()
            except Exception:
                continue
        logger.warning("detect_emotion: no working camera found (tried indices 0-%d)", _MAX_CAMERA_INDEX - 1)
        return None

    def _restore_media(self) -> None:
        """Re-acquire daemon media and restart audio after direct capture."""
        try:
            self._mini.acquire_media()
        except Exception as exc:
            logger.warning("acquire_media failed: %s", exc)
        # Audio recording/playback must be restarted after acquire_media().
        try:
            self._mini.media.start_recording()
        except Exception:
            pass
        try:
            self._mini.media.start_playing()
        except Exception:
            pass
