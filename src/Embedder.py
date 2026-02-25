from sentence_transformers import SentenceTransformer

class Embedder:
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def embed_chunks(self, chunks: list[dict]) -> list[dict]:
        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True
        )

        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()

        return chunks