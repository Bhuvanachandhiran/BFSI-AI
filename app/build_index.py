import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load dataset
with open("../data/bfsidata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

instructions = [item["instruction"] for item in data]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Encode
embeddings = model.encode(instructions)
embeddings = np.array(embeddings).astype("float32")

# Normalize for cosine similarity
faiss.normalize_L2(embeddings)

dimension = embeddings.shape[1]

# Use Inner Product index (cosine after normalization)
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

faiss.write_index(index, "../models/bfsi_index.faiss")

print("Tier 1 cosine index built successfully.")
