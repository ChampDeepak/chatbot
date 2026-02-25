class OutputEvaluator:

    # If best chunk distance exceeds this, retrieval effectively failed
    DISTANCE_THRESHOLD = 1.0

    # Phrases that indicate the LLM refused or couldn't answer
    REFUSAL_PHRASES = [
        "i don't have that information",
        "i don't have information",
        "i cannot help",
        "i can't help",
        "i am unable",
        "i'm unable",
        "i do not have",
        "i don't know",
        "not able to answer",
        "cannot answer",
        "no information available",
        "i cannot provide",
        "i can't provide",
        "outside my knowledge",
        "not in the context",
        "context does not",
        "context doesn't",
    ]

    def evaluate(self, answer: str, distances: list[float], documents: list[str]) -> dict:
        """
        Output Template
        
        Returns:
            {
                "flagged": bool,
                "reasons": list[str],      # why it was flagged
                "label": str               # what to show the user
            }
        """
        reasons = []

        # Check 1: No-context — nothing relevant was retrieved
        no_context = self._check_no_context(distances, documents)
        if no_context:
            reasons.append("no_context")

        # Check 2: Refusal — LLM signaled it couldn't answer
        is_refusal = self._check_refusal(answer)
        if is_refusal:
            reasons.append("refusal")

        flagged = len(reasons) > 0

        return {
            "flagged": flagged,
            "reasons": reasons,
            "label": "⚠️ Low confidence — please verify with support." if flagged else "✓ OK"
        }

    def _check_no_context(self, distances: list[float], documents: list[str]) -> bool:
        # Case A: No documents retrieved at all
        if not documents or len(documents) == 0:
            return True

        # Case B: Best chunk (lowest distance) is still too far away
        best_distance = min(distances)
        if best_distance > self.DISTANCE_THRESHOLD:
            return True

        return False

    def _check_refusal(self, answer: str) -> bool:
        answer_lower = answer.lower().strip()
        return any(phrase in answer_lower for phrase in self.REFUSAL_PHRASES)