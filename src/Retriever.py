import chromadb
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self):
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.client.get_or_create_collection("hr_docs")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def retrieve(self, query: str, doc_name: str = None, n_results: int = 6):
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True
        )[0].tolist()
    
        where_filter = {"document_name": doc_name} if doc_name else None
    
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_filter
        )

        return {
            "documents": results["documents"][0],
            "metadatas": results["metadatas"][0],
            "distances": results["distances"][0]
        }