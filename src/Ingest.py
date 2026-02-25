import os
from Chunk import ChunkData
from Embedder import Embedder
from VectorDB import VectorDB

def ingest_all(text_data_dir: str):
    chunker = ChunkData()
    embedder = Embedder()
    db = VectorDB()

    all_files = [f for f in os.listdir(text_data_dir) if f.endswith(".txt")]
    print(f"Found {len(all_files)} files to ingest\n")

    for filename in all_files:
        file_path = os.path.join(text_data_dir, filename)
        print(f"Processing: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunker.chunk_enterprise_doc(text, filename)
        print(f"  → {len(chunks)} chunks")

        if not chunks:
            print(f"  ⚠ No chunks found, skipping.")
            continue

        chunks_with_embeddings = embedder.embed_chunks(chunks)
        db.store_chunks(chunks_with_embeddings)
        print(f"  ✓ Stored in vector DB")

    print("\nIngestion complete!")

if __name__ == "__main__":
    ingest_all("TextData")