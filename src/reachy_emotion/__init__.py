"""Reachy Emotion — Gemini-powered conversation app for Reachy Mini.

If reachy-mini is not installed, ReachyEmotionApp is a placeholder class that
raises ImportError on instantiation with a clear error message.
"""

from reachy_emotion.main import ReachyEmotionApp
from reachy_emotion.tts_announcer import announce_emotion, speak_text

__all__ = [
    "ReachyEmotionApp",
    "announce_emotion",
    "speak_text",
]
