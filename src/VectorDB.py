import chromadb
import uuid

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("hr_docs")
    
    def sanitize_metadata(self, metadata: dict) -> dict:
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                clean[k] = v
            else:
                clean[k] = str(v)  # fallback: convert to string
        return clean
    
    def store_chunks(self, chunks: list[dict], batch_size=100):

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]

            self.collection.add(
            documents=[c["text"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            # metadatas=[self.sanitize_metadata(c["metadata"]) for c in batch],
            ids=[str(uuid.uuid4()) for _ in batch]
)