"""GeminiBridge: Gemini API chat session with emotion detection as a function tool.

The detect_emotion tool lets Gemini decide when to read the human's emotional
state.  Instead of running local inference, it queries the emotion-cloud gRPC
service via :class:`~reachy_emotion.cloud_client.EmotionCloudClient`, which
streams camera frames continuously in the background and stores the latest
result from the cloud.

Usage::

    client = EmotionCloudClient(mini=mini, endpoint="34.x.x.x:50051")
    client.start()

    bridge = GeminiBridge(api_key="...", cloud_client=client)
    bridge.initialize()
    response_text, emotion_result = bridge.chat("Hello Reachy!")
    bridge.shutdown()

    client.stop()
"""

import logging
from typing import Any

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
    "No sarcasm, no teasing\n"
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
        "Read the human's current emotional state from the emotion-cloud inference "
        "service, which analyses the live camera feed. Returns the dominant emotion, "
        "overall confidence, and derived stress / engagement / arousal metrics."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}


class GeminiBridge:
    """Manages a multi-turn Gemini chat session with emotion detection as a tool call.

    Emotion detection is delegated entirely to the cloud: the bridge holds a
    reference to an :class:`~reachy_emotion.cloud_client.EmotionCloudClient`
    that streams camera frames to emotion-cloud in the background.  When Gemini
    invokes ``detect_emotion`` the bridge just calls
    ``cloud_client.get_latest_result()`` — no local model, no GPU needed on the
    robot.

    All robot *actions* (motion, TTS) are handled by the caller
    (conversation_app.py) after :meth:`chat` returns.
    """

    def __init__(
        self,
        api_key: str,
        cloud_client: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        model: str = DEFAULT_MODEL,
    ) -> None:
        # API key is kept only until initialize() creates the client, then cleared.
        self.__api_key = api_key
        self._cloud_client = cloud_client
        self._system_prompt = system_prompt
        self._model = model
        self._client: Any = None
        self._chat: Any = None
        self._last_emotion_result: dict | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Set up the Gemini client and chat session."""
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=self.__api_key)
        # Clear the key from memory once the client is created.
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

        logger.info("GeminiBridge ready (model=%s)", self._model)

    def shutdown(self) -> None:
        """Release the Gemini client."""
        self._client = None
        self._chat = None
        logger.info("GeminiBridge shutdown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def last_emotion_result(self) -> dict | None:
        """The most recent emotion result dict from this turn, or None."""
        return self._last_emotion_result

    def chat(self, user_text: str) -> tuple[str, dict | None]:
        """Send a user message, handle any tool calls, and return Gemini's reply.

        Args:
            user_text: Transcribed or typed user input.

        Returns:
            Tuple of (Gemini text response, emotion result dict | None).
            The emotion dict is non-None only when Gemini called detect_emotion
            during this turn.

        Raises:
            RuntimeError: If initialize() has not been called.
        """
        if self._chat is None:
            raise RuntimeError("GeminiBridge.initialize() must be called first.")

        from google.genai import types

        self._last_emotion_result = None  # reset per turn

        response = self._chat.send_message(user_text)

        # Tool-call loop: Gemini may call detect_emotion one or more times per turn.
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
        """Fetch the latest emotion result from emotion-cloud and return it.

        Returns a JSON-serialisable dict suitable for a Gemini function response.
        Falls back gracefully if the cloud client has no result yet (buffer
        still warming up) or if the client is unavailable.
        """
        if self._cloud_client is None:
            return {"error": "EmotionCloudClient not available"}

        result = self._cloud_client.get_latest_result()
        if result is None:
            return {
                "dominant_emotion": "unclear",
                "confidence": 0.0,
                "note": "emotion-cloud buffer still warming up — no result yet",
            }

        self._last_emotion_result = result
        return result
