"""conversation_app: core conversation loop for reachy-emotion.

Internal support module — not a separate app.
Entry point is main.py / ReachyEmotionApp.

Flow per turn
─────────────
1. Listen  : record speech from Reachy's mic → transcribe via Google STT
2. Chat    : send text to Gemini (which may call detect_emotion as a tool)
3. React   : if Gemini called detect_emotion, play matching RecordedMove
4. Speak   : TTS Gemini's response through Reachy's speaker

Configuration
─────────────
Set GEMINI_API_KEY     in .env (required).
Set GEMINI_MODEL       in .env (optional, default: gemini-2.5-flash).
Set GEMINI_SYSTEM_PROMPT in .env (optional, single-line override of the default prompt).
"""

import logging
import os
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Maximum characters accepted from text-mode stdin to avoid accidental
# context-window exhaustion when the user pastes large blocks of text.
_MAX_TEXT_INPUT = 2000

# Module-level cache so RecordedMoves is only loaded once per library path.
_recorded_moves_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def _load_env() -> None:
    """Load .env into environment. Warns if python-dotenv is not installed."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        logger.warning(
            "python-dotenv is not installed — .env file will not be loaded. "
            "Run: pip install python-dotenv"
        )


def _load_api_key() -> str:
    """Return GEMINI_API_KEY. Raises ValueError if not set."""
    _load_env()
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. "
            "Add it to a .env file in the project root or export it as an environment variable."
        )
    return key


def _load_model() -> str:
    """Return Gemini model from GEMINI_MODEL env var, or the package default."""
    from reachy_emotion.gemini_bridge import DEFAULT_MODEL
    _load_env()
    return os.environ.get("GEMINI_MODEL", "").strip() or DEFAULT_MODEL


def _load_system_prompt() -> str | None:
    """Return GEMINI_SYSTEM_PROMPT from env if set, otherwise None (use built-in default)."""
    _load_env()
    return os.environ.get("GEMINI_SYSTEM_PROMPT", "").strip() or None


# ---------------------------------------------------------------------------
# Physical emotion reaction
# ---------------------------------------------------------------------------

def _get_recorded_moves(library: str) -> Any | None:
    """Return a cached RecordedMoves instance for *library*, loading it on first call."""
    if library not in _recorded_moves_cache:
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves
            _recorded_moves_cache[library] = RecordedMoves(library)
        except Exception as exc:
            logger.debug("Could not load RecordedMoves('%s'): %s", library, exc)
            _recorded_moves_cache[library] = None
    return _recorded_moves_cache[library]


def _react_to_emotion(emotion_result: Any, mini: Any) -> None:
    """Play a RecordedMove that matches the detected emotion (best-effort)."""
    try:
        from reachy_emotion.reachy_handler import EMOTIONS_LIBRARY

        recorded_moves = _get_recorded_moves(EMOTIONS_LIBRARY)
        if recorded_moves is None:
            return

        moves = recorded_moves.list_moves()
        emotion_label = emotion_result.emotion.dominant_emotion.value.lower()
        match = next((m for m in moves if emotion_label in m.lower()), None)
        if match:
            logger.info("Playing move: %s", match)
            mini.play_move(recorded_moves.get(match), initial_goto_duration=1.0)
    except Exception as exc:
        logger.debug("Emotion reaction skipped: %s", exc)


# ---------------------------------------------------------------------------
# Core conversation loop
# ---------------------------------------------------------------------------

def run_conversation_loop(
    mini: Any,
    stop_event: threading.Event,
    system_prompt: str | None = None,
    voice_mode: bool = True,
    language: str = "en-US",
    model: str | None = None,
) -> None:
    """Drive the listen → Gemini → react → speak cycle until stop_event is set.

    Args:
        mini: Connected ReachyMini instance (owned by caller).
        stop_event: Set to exit the loop cleanly.
        system_prompt: Override the Gemini system prompt. Falls back to
            GEMINI_SYSTEM_PROMPT env var, then the built-in default.
        voice_mode: True = listen via mic; False = read from stdin.
        language: BCP-47 language code for STT/TTS (e.g. "en-US", "fr-FR").
        model: Gemini model name. Falls back to GEMINI_MODEL env / DEFAULT_MODEL.
    """
    from reachy_emotion.gemini_bridge import GeminiBridge, DEFAULT_SYSTEM_PROMPT
    from reachy_emotion.tts_announcer import speak_text
    from reachy_emotion.voice_input import listen

    # Resolve system prompt: explicit arg > env var > built-in default
    resolved_prompt = system_prompt or _load_system_prompt() or DEFAULT_SYSTEM_PROMPT

    bridge = GeminiBridge(
        api_key=_load_api_key(),
        mini=mini,
        system_prompt=resolved_prompt,
        model=model or _load_model(),
    )

    # initialize() is inside the try so shutdown() always runs in finally
    try:
        bridge.initialize()
    except Exception as exc:
        logger.error("Failed to initialise GeminiBridge: %s", exc)
        bridge.shutdown()
        return

    if voice_mode:
        mini.media.start_recording()
        logger.info("Conversation started — speak to Reachy (Ctrl-C to stop)")
    else:
        logger.info("Text mode — type your message (Ctrl-C or Ctrl-D to stop)")

    try:
        while not stop_event.is_set():
            # 1. Input
            if voice_mode:
                user_text = listen(mini, language=language)
                if not user_text:
                    continue
            else:
                try:
                    user_text = input("You: ").strip()
                except EOFError:
                    break
                if not user_text:
                    continue
                if len(user_text) > _MAX_TEXT_INPUT:
                    logger.warning(
                        "Input truncated from %d to %d characters",
                        len(user_text),
                        _MAX_TEXT_INPUT,
                    )
                    user_text = user_text[:_MAX_TEXT_INPUT]

            logger.info("User  → %s", user_text)

            # 2. Gemini (may call detect_emotion tool internally)
            try:
                response_text, emotion_result = bridge.chat(user_text)
            except Exception as exc:
                logger.error("Gemini request failed: %s", exc)
                continue

            if not response_text:
                logger.debug("Empty Gemini response — skipping turn")
                continue

            logger.info("Reachy → %s", response_text)

            # 3. Physical reaction when emotion was read
            if emotion_result is not None:
                _react_to_emotion(emotion_result, mini)

            # 4. Speak response
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
