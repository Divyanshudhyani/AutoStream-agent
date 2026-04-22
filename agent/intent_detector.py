import google.genai as genai
from google.genai import types
import os
import json
import re


INTENT_SYSTEM_PROMPT = """You are an intent classification engine for AutoStream, a SaaS video editing platform.

Your ONLY job is to analyze a user message and return a JSON object with the detected intent.

Classify into EXACTLY ONE of these intents:
1. "greeting"        Casual hello, how are you, small talk, no product question
2. "inquiry"         Asking about features, pricing, plans, refunds, support, or any product question
3. "high_intent"     User expresses desire to sign up, try, buy, subscribe, or start using the product

Rules:
- If the user mentions wanting to try, start, sign up, purchase, subscribe → ALWAYS "high_intent"
- If the user asks what something costs, how something works → "inquiry"  
- If user says hi, hello, hey, thanks → "greeting"
- When in doubt between inquiry and high_intent, choose "high_intent" if user mentions a specific plan

Return ONLY valid JSON, no explanation, no markdown. Example:
{"intent": "high_intent", "confidence": 0.95, "reasoning": "User said they want to try the Pro plan"}
"""


class IntentDetector:
    def __init__(self, api_key: str = None):
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Please set it as an environment variable.")
        
        self.client = genai.Client(api_key=api_key)

    def detect(self, message: str, conversation_history: list = None) -> dict:
        """
        Detect intent from a user message.
        
        Args:
            message: The latest user message
            conversation_history: List of previous messages for context
        
        Returns:
            dict with keys: intent, confidence, reasoning
        """
        # Build context from recent history if available
        context = ""
        if conversation_history:
            recent = conversation_history[-4:]  # Last 4 turns
            context_lines = []
            for turn in recent:
                role = turn.get("role", "")
                content = turn.get("content", "")
                if role in ("user", "assistant"):
                    context_lines.append(f"{role.capitalize()}: {content}")
            if context_lines:
                context = "Recent conversation:\n" + "\n".join(context_lines) + "\n\n"

        # 1. Quick Detection (Instant)
        quick_result = self._quick_detect(message)
        if quick_result:
            return quick_result

        # 2. LLM Detection (Fallback for complex queries)
        prompt = f"{context}Classify this message:\nUser: {message}"

        try:
            response = self.client.models.generate_content(
                model="gemini-flash-lite-latest",
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=INTENT_SYSTEM_PROMPT
                )
            )
            raw = response.text.strip()
            
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            
            result = json.loads(raw)
            # Normalize intent key
            result["intent"] = result.get("intent", "inquiry").lower()
            return result

        except (json.JSONDecodeError, Exception) as e:
            return self._fallback_detect(message, str(e))

    def _quick_detect(self, message: str) -> dict:
        """Instant keyword-based detection for very simple messages."""
        msg = message.lower().strip()
        
        # Simple greetings
        if msg in ["hi", "hello", "hey", "hlo", "hola", "yo", "greetings"]:
            return {"intent": "greeting", "confidence": 1.0, "reasoning": "Simple greeting keyword"}
        
        # Very short high intent
        if msg in ["sign up", "get started", "pricing", "plans"]:
            # "pricing" could be inquiry, but usually leads to high intent flow
            return None # Let LLM decide context for these
            
        return None

    def _fallback_detect(self, message: str, error: str) -> dict:
        """Fallback: simple keyword heuristic."""
        msg_lower = message.lower()
        if any(w in msg_lower for w in ["sign up", "subscribe", "try", "buy", "start", "get", "want"]):
            intent = "high_intent"
        elif any(w in msg_lower for w in ["hi", "hello", "hey", "thanks", "thank you", "hlo"]):
            intent = "greeting"
        else:
            intent = "inquiry"
        
        return {
            "intent": intent,
            "confidence": 0.6,
            "reasoning": f"Fallback classification (LLM error: {error[:50]})"
        }

API_KEY = os.getenv("GEMINI_API_KEY")

def list_available_models():
    genai_client = genai.Client(api_key=API_KEY)
    models = genai_client.models
    print("Available models:")
    try:
        for model in models.list():
            print(f"- {model}")
    except Exception as e:
        print(f"Failed to list models: {e}")
