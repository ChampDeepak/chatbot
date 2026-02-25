# src/RAG.py
import time
from dotenv import load_dotenv
from Retriever import Retriever
from Router import ModelRouter
from groq import Groq

load_dotenv()
client = Groq()
router = ModelRouter()

def answer(query: str, doc_name: str = None) -> dict:
    # Step 1: Route the query BEFORE calling LLM
    decision = router.classify(query)
    model = decision["model"]

    # Step 2: Retrieve chunks
    retriever = Retriever()
    retrieved = retriever.retrieve(query, doc_name=doc_name, n_results=6)
    context = "\n\n---\n\n".join(retrieved["documents"])

    prompt = f"""You are an HR assistant for ClearPath. Answer the employee's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

    # Step 3: Call LLM with routed model + measure latency
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

    # Step 4: Log the routing decision
    router.log(decision, tokens_input, tokens_output, latency_ms)

    return {
        "answer": answer_text,
        "chunks_used": retrieved["documents"],
        "tokens_input": tokens_input,
        "tokens_output": tokens_output,
        "model": model,
        "classification": decision["classification"],
        "routing_reason": decision["reason"],
        "latency_ms": round(latency_ms, 2)
    }

if __name__ == "__main__":
    test_queries = [
        "Hi there!",                                          # → simple
        "Is harassment tolerated at ClearPath?",             # → simple  
        "How do I report a violation?",                      # → complex
        "What is the difference between PTO and sick leave?" # → complex
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        result = answer(q)
        print(f"Model used: {result['model']}")
        print(f"Answer: {result['answer'][:100]}...")
        print("-" * 50)