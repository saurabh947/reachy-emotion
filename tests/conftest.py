"""Pytest configuration for the reachy-emotion test suite.

Stubs out the neural-backbone subpackage of emotion_detection_action
(emotion_detection_action.models.*) that is not included in the lightweight
unit-test install.  The full models package pulls in PyTorch, torchvision,
MediaPipe, and other heavy deps that are not needed for testing action handling
or conversation logic.

This conftest is loaded automatically by pytest before any test file is
collected, so the stubs are in place before any test tries to import
reachy_emotion.reachy_handler (which transitively imports emotion_detection_action).
"""

import sys
from unittest.mock import MagicMock

_STUB_MODULES = [
    "emotion_detection_action.models",
    "emotion_detection_action.models.backbones",
    "emotion_detection_action.models.fusion",
    "emotion_detection_action.models.base",
    "emotion_detection_action.models.registry",
]

for _mod_name in _STUB_MODULES:
    sys.modules.setdefault(_mod_name, MagicMock())
