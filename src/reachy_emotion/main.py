#!/usr/bin/env python3
"""Reachy Emotion — single Gemini-powered conversation app for Reachy Mini.

Reachy listens to you via its microphone, talks with Gemini, and reads your
emotion on-demand when Gemini decides to call the detect_emotion tool.

Dashboard app : ReachyEmotionApp  (registered as "reachy_emotion")
CLI script    : reachy-emotion --help
"""

import argparse
import logging
import sys
import threading

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reachy Mini App Framework class  (dashboard entry point)
# ---------------------------------------------------------------------------

try:
    from reachy_mini import ReachyMini, ReachyMiniApp

    class ReachyEmotionApp(ReachyMiniApp):
        """Gemini-powered conversation app for Reachy Mini.

        Reachy listens, converses with Gemini, and uses the local
        emotion-detection-action SDK as a Gemini tool call when it wants
        to read how you're feeling.

        Configure via .env:
            GEMINI_API_KEY   — required
            GEMINI_MODEL     — optional (default: gemini-2.5-flash)
        """

        custom_app_url: str | None = None

        def run(self, reachy_mini: "ReachyMini", stop_event: threading.Event) -> None:
            from reachy_emotion.system_deps import check_and_warn
            from reachy_emotion.conversation_app import run_conversation_loop, _load_model

            check_and_warn()
            run_conversation_loop(
                mini=reachy_mini,
                stop_event=stop_event,
                model=_load_model(),
            )

except ImportError:
    class ReachyEmotionApp:  # type: ignore[no-redef]
        """Placeholder when reachy-mini SDK is not installed.

        Instantiating this class will raise ImportError with a clear message.
        """

        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ImportError(
                "reachy-mini SDK is not installed. "
                "Install it with: pip install reachy-mini"
            )


# ---------------------------------------------------------------------------
# CLI entry point  (`reachy-emotion` script)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Reachy Emotion: Gemini conversation with on-demand emotion detection"
    )
    parser.add_argument("--sim", action="store_true",
                        help="Simulation mode (start the daemon with --sim first; "
                             "macOS: mjpython -m reachy_mini.daemon.app.main --sim)")
    parser.add_argument("--text", action="store_true",
                        help="Text input instead of voice (useful for testing)")
    parser.add_argument("--lang", default="en-US",
                        help="STT/TTS language code (default: en-US)")
    parser.add_argument("--model", default=None,
                        help="Gemini model name (default: from GEMINI_MODEL env or gemini-2.5-flash)")
    parser.add_argument("--media-backend", default="default",
                        choices=["default", "gstreamer", "webrtc"],
                        help="Reachy media backend")
    parser.add_argument("--prompt", default=None,
                        help="Override Gemini system prompt")
    args = parser.parse_args()

    from reachy_emotion.system_deps import check_and_warn
    from reachy_emotion.conversation_app import run_conversation_loop, _load_model

    check_and_warn()

    try:
        from reachy_mini import ReachyMini
    except ImportError:
        logger.error("reachy-mini is not installed. Run: pip install reachy-mini")
        sys.exit(1)

    if args.sim:
        logger.info("Simulation mode — ensure reachy-mini-daemon --sim is running")

    stop_event = threading.Event()
    with ReachyMini(media_backend=args.media_backend) as mini:
        run_conversation_loop(
            mini=mini,
            stop_event=stop_event,
            system_prompt=args.prompt,
            voice_mode=not args.text,
            language=args.lang,
            model=args.model or _load_model(),
        )


if __name__ == "__main__":
    main()
