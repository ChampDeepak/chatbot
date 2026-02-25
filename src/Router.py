# src/Router.py
import re
import time
import json
from datetime import datetime

class ModelRouter:

    SIMPLE_MODEL = "llama-3.1-8b-instant"
    COMPLEX_MODEL = "llama-3.3-70b-versatile"

    # Signals that strongly indicate complexity
    COMPLEXITY_KEYWORDS = [
        "why", "how", "explain", "describe", "compare", "difference",
        "between", "steps", "process", "policy", "procedure", "what if",
        "should i", "help me", "tell me about", "what happens",
        "when should", "who is responsible", "what are my", "summarize"
    ]

    # Greeting patterns - clearly simple
    GREETING_PATTERNS = [
        "hello", "hi", "hey", "good morning", "good afternoon",
        "good evening", "thanks", "thank you", "bye", "goodbye"
    ]

    # Yes/no question starters
    YES_NO_STARTERS = [
        "is ", "are ", "do ", "does ", "did ", "was ", "were ",
        "can ", "could ", "will ", "would ", "should ", "has ", "have "
    ]

    def classify(self, query: str) -> dict:
        """
        Returns classification dict with model choice and reasoning.
        """
        q = query.strip().lower()
        word_count = len(q.split())
        question_count = query.count("?")

        # Rule 1: Greeting check
        if any(q.startswith(g) or q == g for g in self.GREETING_PATTERNS):
            return self._decision("simple", "greeting detected", query)

        # Rule 2: Yes/No question check (short + starts with yes/no starter)
        if any(q.startswith(s) for s in self.YES_NO_STARTERS) and word_count <= 12:
            return self._decision("simple", "yes/no question", query)

        # Rule 3: Complexity keyword check
        if any(kw in q for kw in self.COMPLEXITY_KEYWORDS):
            return self._decision("complex", "complexity keyword matched", query)

        # Rule 4: Multiple questions → complex
        if question_count > 1:
            return self._decision("complex", "multiple questions detected", query)

        # Rule 5: Long query → complex
        if word_count > 20:
            return self._decision("complex", f"long query ({word_count} words)", query)

        # Rule 6: Very short single-fact lookup → simple
        if word_count <= 8:
            return self._decision("simple", "short single-fact lookup", query)

        # Default: complex (when in doubt, use stronger model)
        return self._decision("complex", "default fallback", query)

    def _decision(self, classification: str, reason: str, query: str) -> dict:
        model = self.SIMPLE_MODEL if classification == "simple" else self.COMPLEX_MODEL
        return {
            "classification": classification,
            "model": model,
            "reason": reason,
            "query": query
        }

    def log(self, decision: dict, tokens_input: int, tokens_output: int, latency_ms: float):
        """
        Logs routing decision in required format.
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": decision["query"],
            "classification": decision["classification"],
            "reason": decision["reason"],
            "model_used": decision["model"],
            "tokens_input": tokens_input,
            "tokens_output": tokens_output,
            "latency_ms": round(latency_ms, 2)
        }
        # Print to console (later you can write to a .jsonl file)
        print(f"[ROUTER LOG] {json.dumps(log_entry, indent=2)}")
        return log_entry