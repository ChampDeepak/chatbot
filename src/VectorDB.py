import chromadb

class VectorDB:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("hr_docs")
    
    def sanitize_metadata(self, metadata: dict) -> dict:
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""  # ChromaDB doesn't like None in metadata
            else:
                clean[k] = str(v)
        return clean
    
    def store_chunks(self, chunks: list[dict], batch_size=100):
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            
            # Deterministic ID = document_name + heading, so re-runs upsert not duplicate
            ids = [
                f"{c['metadata']['document_name']}::{c['metadata']['heading']}::{idx}"
                for idx, c in enumerate(batch, start=i)
            ]
            
            self.collection.upsert(  # upsert not add!
                documents=[c["text"] for c in batch],
                embeddings=[c["embedding"] for c in batch],
                metadatas=[self.sanitize_metadata(c["metadata"]) for c in batch],
                ids=ids
            )