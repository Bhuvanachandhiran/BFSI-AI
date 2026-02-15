import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

rag_folder = "../rag_docs"
documents = []

def chunk_text(text):
    """
    Splits text into chunks based on double newlines to preserve 
    paragraph and section integrity.
    """
    # Split by double newline to capture paragraphs/sections
    sections = text.split("\n\n")
    chunks = []
    for section in sections:
        cleaned = section.strip()
        # Only keep chunks that have enough substance for semantic meaning
        if len(cleaned) > 50:
            chunks.append(cleaned)
    return chunks

# Process files
for filename in os.listdir(rag_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(rag_folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
            # Use the new section-based chunking
            chunks = chunk_text(text)
            for chunk in chunks:
                documents.append(chunk)

print(f"Total semantic chunks created: {len(documents)}")

if len(documents) == 0:
    raise ValueError("No RAG documents found. Check your delimiter or file content.")

# Embedding with Cosine Similarity preparation
embeddings = model.encode(documents)
embeddings = np.array(embeddings).astype("float32")
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

# Save
faiss.write_index(index, "../models/rag_index.faiss")
np.save("../models/rag_texts.npy", documents)

print("RAG index built successfully with logical sections.")