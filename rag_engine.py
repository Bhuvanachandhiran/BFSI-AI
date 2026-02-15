import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

index = faiss.read_index("../models/rag_index.faiss")
documents = np.load("../models/rag_texts.npy", allow_pickle=True)
def retrieve_context(query, threshold=0.55, top_k=3):

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 3)

    top_scores = scores[0]
    top_indices = indices[0]

    max_score = top_scores[0]

    if max_score < threshold:
        return None, max_score

    combined_context = ""
    for idx in top_indices:
        combined_context += documents[idx] + "\n\n"

    return combined_context.strip(), max_score
