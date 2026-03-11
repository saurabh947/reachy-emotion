#!/usr/bin/env python3
"""Reachy Emotion app: detects human emotions and responds with robot actions and speech.

This module exposes ReachyEmotionApp, a ReachyMiniApp subclass that can be installed
and run from the Reachy Mini dashboard. It can also be run directly from the CLI.
"""

import argparse
import logging
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np

from emotion_detection_action import Config, EmotionDetector
from emotion_detection_action.actions.logging_handler import LoggingActionHandler
from emotion_detection_action.core.types import DetectionResult

from reachy_emotion.reachy_handler import ReachyMiniActionHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

GAZE_DURATION = 0.3  # seconds for look_at_image movement


def gaze_at_face(mini: Any, detection: DetectionResult) -> None:
    """Direct Reachy's gaze toward the first detected face in the frame.

    Uses the face bounding box centre as the pixel target for look_at_image().
    No-ops silently if no face is detected or the call fails.

    Args:
        mini: ReachyMini instance.
        detection: DetectionResult containing face bounding boxes.
    """
    if not detection.faces:
        return
    face = detection.faces[0]
    bbox = face.bbox
    u = bbox.x + bbox.width // 2
    v = bbox.y + bbox.height // 2
    try:
        mini.look_at_image(u, v, duration=GAZE_DURATION)
    except Exception as e:
        logger.debug("look_at_image failed: %s", e)


# ---------------------------------------------------------------------------
# Reachy Mini App Framework class
# ---------------------------------------------------------------------------

try:
    from reachy_mini import ReachyMini, ReachyMiniApp

    class ReachyEmotionApp(ReachyMiniApp):
        """Emotion-responsive app for Reachy Mini.

        Detects human emotions via the robot's camera and microphone, then
        responds with pre-recorded movements and spoken announcements.

        Compatible with the Reachy Mini dashboard (start/stop on demand).
        """

        custom_app_url: str | None = None

        def run(self, reachy_mini: "ReachyMini", stop_event: threading.Event) -> None:
            """App entry point called by the Reachy Mini dashboard.

            Args:
                reachy_mini: Already-connected ReachyMini instance provided by the framework.
                stop_event: Set when the user stops the app from the dashboard.
            """
            _run_emotion_loop(
                mini=reachy_mini,
                stop_event=stop_event,
                device="cpu",
                no_audio=False,
                no_announce=False,
                no_gaze=False,
            )

except ImportError:
    # reachy_mini not installed; app class unavailable (CLI mode still works)
    ReachyEmotionApp = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Core loop (shared by app framework and CLI)
# ---------------------------------------------------------------------------

def _run_emotion_loop(
    mini: Any,
    stop_event: threading.Event,
    device: str = "cpu",
    no_audio: bool = False,
    no_announce: bool = False,
    no_gaze: bool = False,
    webcam_id: int | None = None,
) -> None:
    """Main detection-and-response loop.

    Args:
        mini: Connected ReachyMini instance (owned externally).
        stop_event: Loop exits when this is set.
        device: Torch device for emotion models.
        no_audio: Disable microphone input.
        no_announce: Disable TTS speaker output.
        no_gaze: Disable face gaze tracking.
        webcam_id: If set, use this webcam index instead of Reachy's camera.
    """
    handler = ReachyMiniActionHandler(
        mini=mini,
        announce_enabled=not no_announce,
    )
    config = Config(device=device, vla_enabled=False)
    detector = EmotionDetector(config, action_handler=handler)
    detector.initialize()

    if not no_audio:
        try:
            mini.media.start_recording()
        except Exception as e:
            logger.warning("Could not start recording: %s", e)

    cap = None
    if webcam_id is not None:
        cap = cv2.VideoCapture(webcam_id)
        logger.info("Using webcam %s for video input", webcam_id)

    try:
        frame_count = 0
        while not stop_event.is_set():
            if cap is not None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.1)
                    continue
                audio = None
            else:
                frame = mini.media.get_frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                audio_raw = None if no_audio else mini.media.get_audio_sample()
                audio = audio_raw.mean(axis=1).astype(np.float32) if audio_raw is not None else None

            result = detector.process_frame(frame, audio=audio, timestamp=time.time())
            if result:
                if not no_gaze:
                    gaze_at_face(mini, result.detection)

                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(
                        "Emotion: %s | Action: %s",
                        result.emotion.dominant_emotion.value,
                        result.action.action_type,
                    )
    finally:
        if not no_audio:
            try:
                mini.media.stop_recording()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        detector.shutdown()


# ---------------------------------------------------------------------------
# CLI entry point (for running directly without the dashboard)
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini emotion-responsive app")
    parser.add_argument("--sim", action="store_true", help="Simulation mode (run reachy-mini-daemon --sim first)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for emotion models")
    parser.add_argument("--media-backend", default="default", choices=["default", "gstreamer", "webrtc"])
    parser.add_argument("--use-webcam", type=int, default=None, metavar="N", help="Use webcam N instead of Reachy camera")
    parser.add_argument("--no-audio", action="store_true", help="Disable microphone input")
    parser.add_argument("--no-announce", action="store_true", help="Disable TTS emotion announcement via speaker")
    parser.add_argument("--no-gaze", action="store_true", help="Disable face gaze tracking")
    parser.add_argument("--no-robot", action="store_true", help="No Reachy: use webcam + LoggingActionHandler only")
    return parser.parse_args()


def _run_cli_with_reachy(args: argparse.Namespace) -> None:
    """CLI path: create and manage our own ReachyMini connection."""
    if args.sim:
        logger.info("Simulation mode: ensure reachy-mini-daemon --sim is running")

    try:
        from reachy_mini import ReachyMini
    except ImportError:
        logger.error("reachy-mini is not installed. Run: pip install reachy-mini")
        sys.exit(1)

    stop_event = threading.Event()
    mini_obj = ReachyMini(media_backend=args.media_backend)
    with mini_obj as mini:
        _run_emotion_loop(
            mini=mini,
            stop_event=stop_event,
            device=args.device,
            no_audio=args.no_audio,
            no_announce=args.no_announce,
            no_gaze=args.no_gaze,
            webcam_id=args.use_webcam,
        )


def _run_cli_no_robot(args: argparse.Namespace) -> None:
    """CLI path: no Reachy — use webcam + LoggingActionHandler."""
    handler = LoggingActionHandler(verbose=True)
    config = Config(device=args.device, vla_enabled=False)
    detector = EmotionDetector(config, action_handler=handler)
    detector.initialize()

    cam_id = args.use_webcam if args.use_webcam is not None else 0
    cap = cv2.VideoCapture(cam_id)
    if not cap.isOpened():
        logger.error("Could not open webcam %s", cam_id)
        sys.exit(1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            result = detector.process_frame(frame, audio=None, timestamp=time.time())
            if result:
                logger.info(
                    "Emotion: %s | Action: %s",
                    result.emotion.dominant_emotion.value,
                    result.action.action_type,
                )
    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        cap.release()
        detector.shutdown()


def main() -> None:
    args = _parse_args()
    if args.no_robot:
        _run_cli_no_robot(args)
    else:
        _run_cli_with_reachy(args)


if __name__ == "__main__":
    main()
