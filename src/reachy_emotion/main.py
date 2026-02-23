#!/usr/bin/env python3
"""Reachy Mini emotion-responsive app entry point."""

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from emotion_detection_action import Config, EmotionDetector
from emotion_detection_action.actions.logging_handler import LoggingActionHandler

from reachy_emotion.reachy_handler import ReachyMiniActionHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reachy Mini emotion-responsive app")
    parser.add_argument("--sim", action="store_true", help="Use Reachy daemon simulation mode")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Device for emotion models")
    parser.add_argument("--media-backend", default="default", choices=["default", "gstreamer", "webrtc"])
    parser.add_argument("--use-webcam", type=int, default=None, metavar="N", help="Use webcam N instead of Reachy camera (for dev)")
    parser.add_argument("--no-audio", action="store_true", help="Disable microphone input")
    parser.add_argument("--no-announce", action="store_true", help="Disable emotion announcement via speaker")
    parser.add_argument("--no-robot", action="store_true", help="No Reachy: use webcam + LoggingActionHandler")
    return parser.parse_args()


def run_with_reachy(args: argparse.Namespace) -> None:
    """Run with Reachy Mini: SDK Camera + Mic -> Detector -> Handler -> SDK Motion/Speaker."""
    if args.sim:
        logger.info("Simulation mode: start Reachy daemon with --sim first")

    handler = ReachyMiniActionHandler(
        media_backend=args.media_backend,
        announce_enabled=not args.no_announce,
    )
    config = Config(device=args.device, vla_enabled=False)
    detector = EmotionDetector(config, action_handler=handler)
    # initialize() calls handler.connect() which instantiates ReachyMini
    detector.initialize()

    mini = handler.mini
    if not mini:
        logger.error("Failed to connect to Reachy Mini. Is the daemon running? (uv run reachy-mini-daemon%s)", " --sim" if args.sim else "")
        sys.exit(1)

    if not args.no_audio:
        try:
            mini.media.start_recording()
        except Exception as e:
            logger.warning("Could not start recording: %s", e)

    cap = None
    if args.use_webcam is not None:
        cap = cv2.VideoCapture(args.use_webcam)
        logger.info("Using webcam %s", args.use_webcam)

    try:
        frame_count = 0
        while True:
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
                audio_raw = None if args.no_audio else mini.media.get_audio_sample()
                audio = audio_raw.mean(axis=1).astype(np.float32) if audio_raw is not None else None

            result = detector.process_frame(frame, audio=audio, timestamp=time.time())
            if result:
                frame_count += 1
                if frame_count % 30 == 0:
                    logger.info(
                        "Emotion: %s | Action: %s",
                        result.emotion.dominant_emotion.value,
                        result.action.action_type,
                    )

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if not args.no_audio and mini:
            try:
                mini.media.stop_recording()
            except Exception:
                pass
        if cap is not None:
            cap.release()
        detector.shutdown()


def run_no_robot(args: argparse.Namespace) -> None:
    """Run without Reachy: webcam + LoggingActionHandler."""
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
        logger.info("Interrupted by user")
    finally:
        cap.release()
        detector.shutdown()


def main() -> None:
    args = parse_args()
    if args.no_robot:
        run_no_robot(args)
    else:
        run_with_reachy(args)


if __name__ == "__main__":
    main()
