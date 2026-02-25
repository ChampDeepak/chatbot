from Chunk import ChunkData
from Embedder import Embedder 
from VectorDB import VectorDB
from Retriever import Retriever
import os

# file_path = "../TextData/04_Code_of_Conduct_descriptive.txt"
file_path = "/home/deepak/Desktop/chatbot/TextData/04_Code_of_Conduct_descriptive.txt"

if not os.path.exists(file_path):
    print("File not found:", file_path)
else:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunker = ChunkData()
    result = chunker.chunk_enterprise_doc(text, "04_Code_of_Conduct_descriptive.txt")

    # print(result)

    embedder = Embedder()
    chunks_with_embedings = embedder.embed_chunks(result)
    # print(chunks_with_embedings)

    vectorDB = VectorDB()
    vectorDB.store_chunks(chunks_with_embedings)

    retriever = Retriever()
    result = retriever.retrieve("What are expected behavior in ClearPath?")
    print(result)
