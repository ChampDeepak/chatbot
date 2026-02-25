# src/RAG.py
from dotenv import load_dotenv
from Retriever import Retriever
from groq import Groq

load_dotenv()
client = Groq()  # reads GROQ_API_KEY from env

def answer(query: str, doc_name: str = None) -> dict:
    retriever = Retriever()
    retrieved = retriever.retrieve(query, doc_name=doc_name, n_results=6)
    
    # Format chunks into a context block
    context = "\n\n---\n\n".join(retrieved["documents"])
    
    prompt = f"""You are an HR assistant for ClearPath. Answer the employee's question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    
    answer_text = response.choices[0].message.content
    tokens_used = response.usage.total_tokens
    
    return {
        "answer": answer_text,
        "chunks_used": retrieved["documents"],
        "tokens": tokens_used,
        "model": "llama-3.1-8b-instant"
    }

if __name__ == "__main__":
    result = answer("What is the expected behavior at ClearPath?")
    print(result["answer"])
    print(f"\nTokens used: {result['tokens']}")
