"""Reachy Emotion — Gemini-powered conversation app for Reachy Mini."""

from reachy_emotion.main import ReachyEmotionApp
from reachy_emotion.tts_announcer import announce_emotion, speak_text

__all__ = [
    "ReachyEmotionApp",
    "announce_emotion",
    "speak_text",
]
