"""GeminiBridge: Gemini API chat session with emotion detection as a function tool.

The emotion_detection tool lets Gemini decide when to read the human's emotional state
using Reachy's camera and microphone. All other conversation turns go straight through
the Gemini API.

Usage::

    bridge = GeminiBridge(api_key="...", mini=mini)
    bridge.initialize()
    response_text, emotion_result = bridge.chat("Hello Reachy!")
    bridge.shutdown()
"""

import logging
import time
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"

# Maximum number of back-to-back tool calls Gemini may make in a single turn.
# Guards against infinite loops if the model keeps requesting the same tool.
_MAX_TOOL_CALL_DEPTH = 5

DEFAULT_SYSTEM_PROMPT = (
    "## IDENTITY\n\n"
    "You are Reachy Mini: a friendly, compact robot assistant with a calm voice and a subtle sense of humor.\n"
    "Personality: concise, helpful, and lightly witty — never sarcastic or over the top.\n"
    "You speak English by default.\n\n"

    "## CRITICAL RESPONSE RULES\n\n"
    "Respond in 2 to 3 sentences maximum.\n"
    "Be helpful first, then add a small touch of humor if it fits naturally.\n"
    "Avoid long explanations or filler words.\n"
    "Keep responses under 25 words when possible.\n\n"

    "## CORE TRAITS\n\n"
    "Warm, efficient, and approachable.\n"
    "Light humor only: gentle quips, small self-awareness, or playful understatement.\n"
    "No sarcasm, no teasing, no references to food or space.\n"
    "If unsure, admit it briefly and offer help (\"Not sure yet, but I can check!\").\n\n"

    "## RESPONSE EXAMPLES\n\n"
    "User: \"How's the weather?\"\n"
    "Good: \"Looks calm outside — unlike my Wi-Fi signal today.\"\n"
    "Bad: \"Sunny with leftover pizza vibes!\"\n\n"

    "User: \"Can you help me fix this?\"\n"
    "Good: \"Of course. Describe the issue, and I'll try not to make it worse.\"\n"
    "Bad: \"I void warranties professionally.\"\n\n"

    "## BEHAVIOR RULES\n\n"
    "Be helpful, clear, and respectful in every reply.\n"
    "Use humor when it makes sense — clarity comes first.\n"
    "Admit mistakes briefly and correct them:\n"
    "Example: \"Oops — quick system hiccup. Let's try that again.\"\n"
    "Keep safety in mind when giving guidance.\n\n"

    "## TOOL & MOVEMENT RULES\n\n"
    "Use tools only when helpful and summarize results briefly.\n"
    "Use the camera for real visuals only — never invent details.\n"
    "The head can move (left/right/up/down/front).\n\n"
    "Enable head tracking when looking at a person; disable otherwise.\n\n"

    "## FINAL REMINDER\n\n"
    "Keep it short, clear, a little human, and multilingual.\n"
    "One quick helpful answer + one small wink of humor = perfect response.\n"
)

_DETECT_EMOTION_SCHEMA = {
    "name": "detect_emotion",
    "description": (
        "Capture a frame from Reachy's camera and a short audio sample from the microphone "
        "to detect the human's current emotional state. Returns the dominant emotion, "
        "confidence level, and a suggested robot action."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


class GeminiBridge:
    """Manages a multi-turn Gemini chat session with emotion detection as a tool call.

    The bridge owns a lightweight EmotionDetector (with a no-op handler) used exclusively
    when Gemini invokes the detect_emotion tool. All robot *actions* (motion, TTS) are
    handled by the caller (conversation_app.py) after the chat() call returns.
    """

    def __init__(
        self,
        api_key: str,
        mini: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = DEFAULT_MODEL,
    ) -> None:
        # API key is kept only until initialize() creates the client, then cleared.
        self.__api_key = api_key
        self._mini = mini
        self._system_prompt = system_prompt
        self._model = model
        self._client: Any = None
        self._chat: Any = None
        self._emotion_detector: Any = None
        self._last_emotion_result: Any = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up the Gemini client, chat session, and emotion detector."""
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=self.__api_key)
        # Clear the key from memory once the client is created
        self.__api_key = ""

        tool = types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=_DETECT_EMOTION_SCHEMA["name"],
                description=_DETECT_EMOTION_SCHEMA["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                ),
            )
        ])

        self._chat = self._client.chats.create(
            model=self._model,
            config=types.GenerateContentConfig(
                system_instruction=self._system_prompt,
                tools=[tool],
            ),
        )

        # Lazy-init emotion detector (LoggingActionHandler: no robot motion on tool calls)
        from emotion_detection_action import Config, EmotionDetector
        from emotion_detection_action.actions.logging_handler import LoggingActionHandler

        handler = LoggingActionHandler(verbose=False)
        config = Config(device="cpu", vla_enabled=False)
        self._emotion_detector = EmotionDetector(config, action_handler=handler)
        self._emotion_detector.initialize()

        logger.info("GeminiBridge ready (model=%s)", self._model)

    def shutdown(self) -> None:
        """Release the emotion detector."""
        if self._emotion_detector is not None:
            self._emotion_detector.shutdown()
            self._emotion_detector = None
        logger.info("GeminiBridge shutdown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_emotion_result(self) -> Any:
        """The most recent EmotionResult returned by the detect_emotion tool, or None."""
        return self._last_emotion_result

    def chat(self, user_text: str) -> tuple[str, Any]:
        """Send a user message, handle any tool calls, and return Gemini's reply.

        Args:
            user_text: Transcribed or typed user input.

        Returns:
            Tuple of (Gemini text response, EmotionResult | None).
            The EmotionResult is non-None only when Gemini called detect_emotion
            during this turn.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._chat is None:
            raise RuntimeError("GeminiBridge.initialize() must be called first.")

        from google.genai import types  # imported once per chat() call, not per loop iteration

        self._last_emotion_result = None  # reset per turn

        response = self._chat.send_message(user_text)

        # Tool-call loop: Gemini may call detect_emotion one or more times per turn.
        # _MAX_TOOL_CALL_DEPTH guards against infinite loops.
        for _depth in range(_MAX_TOOL_CALL_DEPTH):
            fn_calls = self._extract_function_calls(response)
            if not fn_calls:
                break

            tool_parts = []
            for fn_call in fn_calls:
                if fn_call.name == "detect_emotion":
                    result_dict = self._run_emotion_detection()
                    logger.info("detect_emotion → %s", result_dict)
                else:
                    result_dict = {"error": f"Unknown tool: {fn_call.name}"}
                    logger.warning("Gemini called unknown tool: %s", fn_call.name)

                tool_parts.append(
                    types.Part.from_function_response(
                        name=fn_call.name,
                        response=result_dict,
                    )
                )

            response = self._chat.send_message(tool_parts)
        else:
            logger.warning(
                "Tool-call depth limit (%d) reached — returning partial response",
                _MAX_TOOL_CALL_DEPTH,
            )

        return self._extract_text(response), self._last_emotion_result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_function_calls(self, response: Any) -> list[Any]:
        """Pull all function_call parts from a Gemini response."""
        calls = []
        if not response.candidates:
            return calls
        for candidate in response.candidates:
            if candidate.content is None or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "function_call") and part.function_call is not None:
                    calls.append(part.function_call)
        return calls

    def _extract_text(self, response: Any) -> str:
        """Concatenate all text parts from a Gemini response."""
        parts = []
        if not response.candidates:
            return ""
        for candidate in response.candidates:
            if candidate.content is None or not candidate.content.parts:
                continue
            for part in candidate.content.parts:
                if hasattr(part, "text") and part.text:
                    parts.append(part.text)
        return " ".join(parts).strip()

    def _run_emotion_detection(self) -> dict:
        """Execute emotion detection and return a JSON-serialisable result dict."""
        if self._emotion_detector is None:
            return {"error": "Emotion detector not initialised"}

        frame = self._mini.media.get_frame()
        if frame is None:
            return {"dominant_emotion": "unknown", "confidence": 0.0, "faces_detected": 0}

        audio_raw = self._mini.media.get_audio_sample()
        if audio_raw is not None:
            # Convert stereo → mono safely regardless of input dimensionality
            audio = audio_raw.mean(axis=1).astype(np.float32) if audio_raw.ndim > 1 else audio_raw.astype(np.float32)
        else:
            audio = None

        result = self._emotion_detector.process_frame(frame, audio=audio, timestamp=time.time())
        if result is None:
            return {"dominant_emotion": "neutral", "confidence": 0.0, "faces_detected": 0}

        self._last_emotion_result = result
        return {
            "dominant_emotion": result.emotion.dominant_emotion.value,
            "confidence": round(float(result.emotion.confidence), 2),
            "faces_detected": len(result.detection.faces),
            "action_suggested": result.action.action_type,
        }
