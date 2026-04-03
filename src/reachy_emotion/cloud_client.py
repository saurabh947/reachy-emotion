"""EmotionCloudClient: gRPC streaming client for the emotion-cloud inference service.

emotion-cloud runs on Google Cloud Kubernetes (GKE) and performs neural emotion
detection using a Two-Tower Multimodal Transformer (ViT-B/16 video + emotion2vec
audio). This module maintains a persistent bidirectional gRPC stream to that
service, continuously feeding camera frames and storing the latest result.

GeminiBridge's detect_emotion tool reads from that result store — Gemini decides
*when* to look, but inference runs continuously in the background so the answer
is always fresh.

Usage::

    client = EmotionCloudClient(mini=mini, endpoint="34.x.x.x:50051")
    client.start()

    result = client.get_latest_result()   # None until buffer warms up (~16 frames)
    if result:
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

import numpy as np

logger = logging.getLogger(__name__)

# Target frame rate for streaming (fps). emotion-cloud expects 10–30 fps.
_STREAM_FPS = 15
_FRAME_INTERVAL = 1.0 / _STREAM_FPS

# Seconds to wait before reconnecting after a gRPC stream error.
_RECONNECT_DELAY_SEC = 2.0


# ---------------------------------------------------------------------------
# Proto stub loader
# ---------------------------------------------------------------------------

def _load_stubs() -> tuple[Any, Any]:
    """Return (emotion_pb2, emotion_pb2_grpc), generating stubs if needed.

    Stubs are generated once from proto/emotion.proto using grpc_tools.protoc
    and cached in the proto/ directory alongside the .proto source. They are
    committed to .gitignore so they never land in version control but survive
    across Python sessions.
    """
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

    # Add proto_dir to sys.path so the generated import inside emotion_pb2_grpc
    # (`import emotion_pb2 as emotion__pb2`) resolves correctly.
    proto_dir_str = str(proto_dir)
    if proto_dir_str not in sys.path:
        sys.path.insert(0, proto_dir_str)

    def _dynamic_import(module_name: str, file_path: Path) -> Any:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod  # register before exec so cross-imports work
        spec.loader.exec_module(mod)
        return mod

    pb2 = _dynamic_import("emotion_pb2", pb2_path)
    pb2_grpc = _dynamic_import("emotion_pb2_grpc", proto_dir / "emotion_pb2_grpc.py")
    return pb2, pb2_grpc


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class EmotionCloudClient:
    """Streams Reachy camera frames to emotion-cloud and exposes the latest result.

    The client runs a background daemon thread that:
      1. Opens a bidirectional gRPC stream to emotion-cloud.
      2. Continuously captures frames (and optional audio) from the robot.
      3. Sends them at ``_STREAM_FPS`` fps.
      4. Stores each non-buffering EmotionResponse as a plain dict.

    GeminiBridge calls :meth:`get_latest_result` when Gemini invokes the
    ``detect_emotion`` tool.  The result is always the most recent inference
    from the cloud — no additional latency on the tool-call path.

    Args:
        mini: Connected ReachyMini instance (camera + mic access).
        endpoint: emotion-cloud gRPC address, e.g. ``"34.x.x.x:50051"``.
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
        self._latest_result: dict | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._pb2: Any = None
        self._pb2_grpc: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Load proto stubs and launch the background streaming thread."""
        self._pb2, self._pb2_grpc = _load_stubs()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._stream_loop,
            name="emotion-cloud-stream",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "EmotionCloudClient started (endpoint=%s, session=%s)",
            self._endpoint,
            self._session_id,
        )

    def stop(self) -> None:
        """Signal the streaming thread to exit and wait for it to finish."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("EmotionCloudClient stopped")

    def health_check(self, timeout: float = 5.0) -> dict:
        """Perform a synchronous HealthCheck RPC against emotion-cloud.

        Args:
            timeout: Seconds to wait for a response before raising.

        Returns:
            Dict with keys: ``healthy`` (bool), ``model_status`` (str),
            ``active_sessions`` (int).

        Raises:
            RuntimeError: If start() has not been called (stubs not loaded).
            Exception: On any gRPC connectivity error.
        """
        if self._pb2 is None or self._pb2_grpc is None:
            raise RuntimeError("EmotionCloudClient.start() must be called before health_check()")

        import grpc

        channel = grpc.insecure_channel(
            self._endpoint,
            options=[("grpc.max_receive_message_length", 1024 * 1024)],
        )
        try:
            stub = self._pb2_grpc.EmotionDetectionStub(channel)
            response = stub.HealthCheck(
                self._pb2.HealthRequest(),
                timeout=timeout,
            )
            return {
                "healthy": response.healthy,
                "model_status": response.model_status,
                "active_sessions": response.active_sessions,
            }
        finally:
            channel.close()

    # ------------------------------------------------------------------
    # Public API (called by GeminiBridge on the main thread)
    # ------------------------------------------------------------------

    def get_latest_result(self) -> dict | None:
        """Return the most recent emotion result dict, or None if not yet available.

        Returns a dict with keys:
            dominant_emotion (str): e.g. "happy", "neutral"
            confidence (float): overall softmax confidence in [0, 1]
            confidence_scores (dict[str, float]): per-class probabilities
            stress (float): derived stress metric in [0, 1]
            engagement (float): derived engagement metric in [0, 1]
            arousal (float): derived arousal metric in [0, 1]
        """
        with self._lock:
            return self._latest_result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request_iterator(self):
        """Generator that yields EmotionRequest messages paced at _STREAM_FPS."""
        while not self._stop_event.is_set():
            frame_start = time.monotonic()

            try:
                frame = self._mini.media.get_frame()
                if frame is None:
                    time.sleep(_FRAME_INTERVAL)
                    continue

                # Reachy camera returns BGR; emotion-cloud expects RGB.
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame_rgb = frame[:, :, ::-1].copy()
                else:
                    frame_rgb = frame

                h, w = frame_rgb.shape[:2]

                # Best-effort audio: mono float32 PCM at 16 kHz.
                audio_bytes = b""
                audio_rate = 16000
                try:
                    raw_audio = self._mini.media.get_audio_sample()
                    if raw_audio is not None:
                        mono = (
                            raw_audio.mean(axis=1)
                            if raw_audio.ndim > 1
                            else raw_audio
                        )
                        audio_bytes = mono.astype(np.float32).tobytes()
                except Exception:
                    pass  # audio is optional; model runs video-only if absent

                yield self._pb2.EmotionRequest(
                    session_id=self._session_id,
                    video_frame=frame_rgb.astype(np.uint8).tobytes(),
                    frame_width=w,
                    frame_height=h,
                    audio_chunk=audio_bytes,
                    audio_sample_rate=audio_rate,
                    timestamp_ms=int(time.time() * 1000),
                )

            except Exception as exc:
                logger.debug("Frame capture error (skipping frame): %s", exc)

            # Pace to target FPS regardless of how long capture took.
            elapsed = time.monotonic() - frame_start
            sleep_for = _FRAME_INTERVAL - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _stream_loop(self) -> None:
        """Background thread: open gRPC channel and iterate over responses.

        Reconnects automatically on error with exponential-ish back-off
        (fixed _RECONNECT_DELAY_SEC for now — sufficient for LAN/K8s).
        """
        import grpc  # imported here so the module can be imported without grpcio installed

        while not self._stop_event.is_set():
            channel = None
            try:
                channel = grpc.insecure_channel(
                    self._endpoint,
                    options=[
                        ("grpc.keepalive_time_ms", 30_000),
                        ("grpc.keepalive_timeout_ms", 10_000),
                        ("grpc.max_receive_message_length", 10 * 1024 * 1024),
                    ],
                )
                stub = self._pb2_grpc.EmotionDetectionStub(channel)
                logger.debug("gRPC stream open → %s", self._endpoint)

                for response in stub.StreamEmotion(self._request_iterator()):
                    if self._stop_event.is_set():
                        break

                    if response.error:
                        logger.warning("emotion-cloud error: %s", response.error)
                        continue

                    if response.buffering:
                        logger.debug("emotion-cloud: warming 16-frame buffer …")
                        continue

                    with self._lock:
                        self._latest_result = {
                            "dominant_emotion": response.dominant_emotion,
                            "confidence": round(response.overall_confidence, 2),
                            "confidence_scores": dict(response.confidence_scores),
                            "stress": round(response.stress, 2),
                            "engagement": round(response.engagement, 2),
                            "arousal": round(response.arousal, 2),
                        }

            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "emotion-cloud stream error: %s — reconnecting in %ds",
                    exc,
                    _RECONNECT_DELAY_SEC,
                )
                time.sleep(_RECONNECT_DELAY_SEC)
            finally:
                if channel is not None:
                    try:
                        channel.close()
                    except Exception:
                        pass
