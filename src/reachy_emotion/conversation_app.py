"""ReachyConversationApp: Gemini-powered conversational mode for Reachy Mini.

Flow per turn
─────────────
1. Listen: record speech from Reachy's mic → transcribe via Google STT
2. Chat:   send text to Gemini (which may call detect_emotion as a tool)
3. React:  if Gemini called detect_emotion, play matching RecordedMove
4. Speak:  TTS Gemini's response through Reachy's speaker

Configuration
─────────────
Set GEMINI_API_KEY in a .env file (or as an environment variable).
Optionally set GEMINI_SYSTEM_PROMPT to customise Reachy's personality.
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_api_key() -> str:
    """Load GEMINI_API_KEY from .env or environment. Raises ValueError if missing."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # python-dotenv optional; fall back to raw env

    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to a .env file in the project root or export it as an environment variable."
        )
    return key


def _react_to_emotion(emotion_result: Any, mini: Any) -> None:
    """Play a RecordedMove matching the detected emotion (best-effort, non-blocking)."""
    try:
        from reachy_mini.motion.recorded_move import RecordedMoves
        from reachy_emotion.reachy_handler import EMOTIONS_LIBRARY

        recorded_moves = RecordedMoves(EMOTIONS_LIBRARY)
        moves = recorded_moves.list_moves()
        emotion_label = emotion_result.emotion.dominant_emotion.value.lower()
        match = next((m for m in moves if emotion_label in m.lower()), None)
        if match:
            logger.info("Playing move: %s", match)
            recorded_moves.play(match)
    except Exception as exc:
        logger.debug("Emotion reaction skipped: %s", exc)


# ---------------------------------------------------------------------------
# Core conversation loop (shared by app framework and CLI)
# ---------------------------------------------------------------------------

def run_conversation_loop(
    mini: Any,
    stop_event: threading.Event,
    system_prompt: str | None = None,
    voice_mode: bool = True,
    language: str = "en-US",
    model: str = "gemini-2.0-flash",
) -> None:
    """Drive the listen → Gemini → react → speak cycle until stop_event is set.

    Args:
        mini: ReachyMini instance (already connected).
        stop_event: Set this to exit the loop cleanly.
        system_prompt: Custom Gemini system prompt. Defaults to DEFAULT_SYSTEM_PROMPT.
        voice_mode: If True, listen via Reachy's mic; if False, read from stdin.
        language: STT language code (e.g. "en-US", "fr-FR").
        model: Gemini model name.
    """
    from reachy_emotion.gemini_bridge import GeminiBridge, DEFAULT_SYSTEM_PROMPT
    from reachy_emotion.tts_announcer import speak_text
    from reachy_emotion.voice_input import listen

    api_key = _load_api_key()
    bridge = GeminiBridge(
        api_key=api_key,
        mini=mini,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        model=model,
    )
    bridge.initialize()

    if voice_mode:
        mini.media.start_recording()
        logger.info("Conversation started — speak to Reachy (Ctrl-C to stop)")
    else:
        logger.info("Text conversation started — type your message (Ctrl-C to stop)")

    try:
        while not stop_event.is_set():
            # --- Input ---
            if voice_mode:
                user_text = listen(mini, language=language)
                if not user_text:
                    continue  # nothing heard; loop again
            else:
                try:
                    user_text = input("You: ").strip()
                except EOFError:
                    break
                if not user_text:
                    continue

            logger.info("User → %s", user_text)

            # --- Gemini ---
            try:
                response_text, emotion_result = bridge.chat(user_text)
            except Exception as exc:
                logger.error("Gemini request failed: %s", exc)
                continue

            if not response_text:
                logger.debug("Empty Gemini response — skipping")
                continue

            logger.info("Reachy → %s", response_text)

            # --- Physical reaction (if Gemini read emotion) ---
            if emotion_result is not None:
                _react_to_emotion(emotion_result, mini)

            # --- Speak ---
            speak_text(response_text, mini, lang=language.split("-")[0])

    except KeyboardInterrupt:
        logger.info("Conversation interrupted")
    finally:
        if voice_mode:
            try:
                mini.media.stop_recording()
            except Exception:
                pass
        bridge.shutdown()


# ---------------------------------------------------------------------------
# Reachy Mini App Framework class
# ---------------------------------------------------------------------------

try:
    from reachy_mini import ReachyMini, ReachyMiniApp

    class ReachyConversationApp(ReachyMiniApp):
        """Gemini-powered conversational app for the Reachy Mini dashboard.

        Install GEMINI_API_KEY in the robot's .env before starting.
        """

        custom_app_url: str | None = None

        def run(self, reachy_mini: "ReachyMini", stop_event: threading.Event) -> None:
            run_conversation_loop(mini=reachy_mini, stop_event=stop_event)

except ImportError:
    ReachyConversationApp = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# CLI entry point  (`reachy-emotion-chat` script)
# ---------------------------------------------------------------------------

def _cli_main() -> None:
    """CLI wrapper: `reachy-emotion-chat [--sim] [--text] [--lang en-US] [--model gemini-2.0-flash]`."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Reachy Mini Gemini conversation app")
    parser.add_argument("--sim", action="store_true", help="Simulation mode")
    parser.add_argument("--text", action="store_true", help="Text input instead of voice")
    parser.add_argument("--lang", default="en-US", help="STT language (default: en-US)")
    parser.add_argument("--model", default="gemini-2.0-flash", help="Gemini model name")
    parser.add_argument("--media-backend", default="default", help="Reachy media backend")
    parser.add_argument("--prompt", default=None, help="Custom system prompt text")
    args = parser.parse_args()

    try:
        from reachy_mini import ReachyMini
    except ImportError:
        print("reachy-mini is not installed. Run: pip install reachy-mini", file=sys.stderr)
        sys.exit(1)

    stop_event = threading.Event()
    with ReachyMini(media_backend=args.media_backend) as mini:
        run_conversation_loop(
            mini=mini,
            stop_event=stop_event,
            system_prompt=args.prompt,
            voice_mode=not args.text,
            language=args.lang,
            model=args.model,
        )
