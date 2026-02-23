"""ReachyMiniActionHandler: receives ActionCommand, invokes ReachyMini SDK Motion, RecordedMoves, TTS."""

import logging
import threading
import time
from typing import Any

import numpy as np

from emotion_detection_action.actions.base import BaseActionHandler
from emotion_detection_action.core.types import ActionCommand

from reachy_emotion.tts_announcer import announce_emotion

logger = logging.getLogger(__name__)

EMOTIONS_LIBRARY = "pollen-robotics/reachy-mini-emotions-library"
ANNOUNCE_INTERVAL_SEC = 5.0


class ReachyMiniActionHandler(BaseActionHandler):
    """Action handler that maps ActionCommand to Reachy Mini SDK commands."""

    def __init__(
        self,
        media_backend: str = "default",
        announce_enabled: bool = True,
        announce_interval: float = ANNOUNCE_INTERVAL_SEC,
        name: str = "reachy_mini",
    ) -> None:
        super().__init__(name=name)
        self._media_backend = media_backend
        self._announce_enabled = announce_enabled
        self._announce_interval = announce_interval
        self._mini: Any = None
        self._recorded_moves: Any = None
        self._last_announce_emotion: str | None = None
        self._last_announce_time: float = 0.0
        self._last_action_time: float = 0.0
        self._action_interval: float = 2.0  # Throttle actions

    @property
    def mini(self):
        """ReachyMini instance for main loop to access Camera and Mic."""
        return self._mini

    def connect(self) -> bool:
        try:
            from reachy_mini import ReachyMini
            from reachy_mini.motion.recorded_move import RecordedMoves

            mini = ReachyMini(media_backend=self._media_backend)
            self._mini = mini.__enter__()  # Enter context manager to activate connection
            self._recorded_moves = RecordedMoves(EMOTIONS_LIBRARY)
            self._is_connected = True
            logger.info("ReachyMiniActionHandler connected")
            return True
        except Exception as e:
            logger.error("Failed to connect ReachyMini: %s", e)
            return False

    def disconnect(self) -> None:
        if self._mini is not None:
            try:
                self._mini.__exit__(None, None, None)
            except Exception:
                pass
        self._mini = None
        self._recorded_moves = None
        self._is_connected = False
        logger.info("ReachyMiniActionHandler disconnected")

    def _get_available_moves(self) -> list[str]:
        if not self._recorded_moves:
            return []
        try:
            return list(self._recorded_moves.list_moves())
        except Exception:
            return []

    def _emotion_to_move(self, emotion: str) -> str | None:
        """Map emotion label to RecordedMoves move name."""
        available = self._get_available_moves()
        emotion_lower = emotion.lower() if emotion else ""
        # Direct match
        if emotion_lower in available:
            return emotion_lower
        # Common aliases
        aliases = {
            "happy": ["happy", "joy", "smile"],
            "sad": ["sad", "sadness", "crying"],
            "angry": ["angry", "anger"],
            "fearful": ["fearful", "fear", "scared"],
            "surprised": ["surprised", "surprise"],
            "disgusted": ["disgusted", "disgust"],
            "neutral": ["neutral", "calm"],
        }
        for candidate in aliases.get(emotion_lower, [emotion_lower]):
            if candidate in available:
                return candidate
        return available[0] if available else None

    def _execute_motion(self, action: ActionCommand) -> bool:
        """Execute physical action via SDK Motion and RecordedMoves.

        Note: announcement is handled separately in execute() to avoid double-triggering.
        """
        if not self._mini:
            return False
        try:
            action_type = action.action_type
            params = action.parameters or {}
            emotion = params.get("emotion", "neutral")

            if action_type == "idle":
                self._mini.goto_target(antennas=np.array([0.0, 0.0]), duration=0.5)
            elif action_type == "acknowledge":
                move_name = self._emotion_to_move("happy")
                if move_name:
                    self._mini.play_move(self._recorded_moves.get(move_name), initial_goto_duration=1.0)
                else:
                    self._mini.goto_target(antennas=np.array([0.5, 0.5]), duration=0.5)
            elif action_type == "comfort":
                move_name = self._emotion_to_move("sad")
                if move_name:
                    self._mini.play_move(self._recorded_moves.get(move_name), initial_goto_duration=1.0)
                else:
                    self._mini.goto_target(antennas=np.array([0.2, 0.2]), duration=0.8)
            elif action_type == "de_escalate":
                self._mini.goto_target(body_yaw=np.deg2rad(-15), duration=1.0)
            elif action_type == "reassure":
                move_name = self._emotion_to_move("fearful")
                if move_name:
                    self._mini.play_move(self._recorded_moves.get(move_name), initial_goto_duration=1.0)
                else:
                    self._mini.goto_target(antennas=np.array([0.3, 0.3]), duration=0.8)
            elif action_type == "wait":
                pass  # No-op: hold current pose
            elif action_type == "retreat":
                self._mini.goto_target(body_yaw=np.deg2rad(-25), duration=1.0)
            elif action_type == "approach":
                self._mini.goto_target(body_yaw=np.deg2rad(15), duration=1.0)
            elif action_type == "speak":
                pass  # Announcement handled in execute(); no additional motion
            elif action_type in ("gesture", "stub"):
                move_name = self._emotion_to_move(emotion)
                if move_name:
                    self._mini.play_move(self._recorded_moves.get(move_name), initial_goto_duration=1.0)
                else:
                    self._mini.goto_target(antennas=np.array([0.3, 0.3]), duration=0.5)
            else:
                # Unknown: try emotion-based RecordedMove, fall back to neutral
                move_name = self._emotion_to_move(emotion)
                if move_name:
                    self._mini.play_move(self._recorded_moves.get(move_name), initial_goto_duration=1.0)
                else:
                    self._mini.goto_target(antennas=np.array([0.0, 0.0]), duration=0.5)

            return True
        except Exception as e:
            logger.warning("Motion execution failed: %s", e)
            return False

    def _do_announce(self, emotion: str) -> None:
        """Announce emotion via TTS (with throttling)."""
        now = time.time()
        if emotion != self._last_announce_emotion or (now - self._last_announce_time) >= self._announce_interval:
            self._last_announce_emotion = emotion
            self._last_announce_time = now
            threading.Thread(target=announce_emotion, args=(emotion, self._mini), daemon=True).start()

    def get_supported_actions(self) -> list[str]:
        return super().get_supported_actions() + ["stub"]

    def execute(self, action: ActionCommand) -> bool:
        if not self._is_connected or not self._mini:
            return False
        now = time.time()
        if now - self._last_action_time < self._action_interval:
            return True  # Throttled
        self._last_action_time = now

        valid, msg = self.validate_action(action)
        if not valid:
            logger.debug("Invalid action: %s", msg)
            return False

        success = self._execute_motion(action)

        # Announce emotion once via TTS → SDK Speaker (throttled)
        if success and self._announce_enabled:
            emotion = (action.parameters or {}).get("emotion", "neutral")
            self._do_announce(emotion)

        return success
