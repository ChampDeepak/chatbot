# src/RAG.py
import time
from dotenv import load_dotenv
from Retriever import Retriever
from Router import ModelRouter
from Evaluator import OutputEvaluator
from groq import Groq

load_dotenv()
client = Groq()
router = ModelRouter()
evaluator = OutputEvaluator()

def answer(query: str, doc_name: str = None) -> dict:
    # Step 1: Route
    decision = router.classify(query)
    model = decision["model"]

    # Step 2: Retrieve
    retriever = Retriever()
    retrieved = retriever.retrieve(query, doc_name=doc_name, n_results=6)
    context = "\n\n---\n\n".join(retrieved["documents"])

    prompt = f"""You are an HR assistant for ClearPath. Answer the employee's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

    # Step 3: Call LLM
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    latency_ms = (time.time() - start) * 1000

    answer_text = response.choices[0].message.content
    tokens_input = response.usage.prompt_tokens
    tokens_output = response.usage.completion_tokens

    # Step 4: Evaluate output  ← NEW
    evaluation = evaluator.evaluate(
        answer=answer_text,
        distances=retrieved["distances"],
        documents=retrieved["documents"]
    )

    # Step 5: Log
    router.log(decision, tokens_input, tokens_output, latency_ms)

    return {
        "answer": answer_text,
        "chunks_used": retrieved["documents"],
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "model": model,
        "classification": decision["classification"],
        "routing_reason": decision["reason"],
        "latency_ms": round(latency_ms, 2),
        # Evaluation fields
        "flagged": evaluation["flagged"],
        "flag_reasons": evaluation["reasons"],
        "confidence_label": evaluation["label"],   # ← this goes to the UI
    }

if __name__ == "__main__":
    test_cases = [
        # Should be OK — relevant chunks exist
        "What is the expected behavior at ClearPath?",
        
        # Should flag no_context — totally off-topic
        "What is the capital of France?",
        
        # Should flag refusal — vague enough LLM might say it doesn't know
        "Explain the quantum entanglement policy",
    ]
    for q in test_cases:
        print(f"\nQuery: {q}")
        result = answer(q)
        print(f"Model used: {result['model']}")
        print(f"Answer: {result['answer'][:100]}...")
        print("-" * 50)
        # print(result)

# if __name__ == "__main__":
#     test_queries = [
#         "Hi there!",                                          # → simple
#         "Is harassment tolerated at ClearPath?",             # → simple  
#         "How do I report a violation?",                      # → complex
#         "What is the difference between PTO and sick leave?" # → complex
#     ]

#     for q in test_queries:
#         print(f"\nQuery: {q}")
#         result = answer(q)
#         print(f"Model used: {result['model']}")
#         print(f"Answer: {result['answer'][:100]}...")
#         print("-" * 50)