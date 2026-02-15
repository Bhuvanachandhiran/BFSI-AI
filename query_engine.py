import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from slm_engine import generate_response
from rag_engine import retrieve_context

# -------------------- LOAD DATA --------------------

with open("../data/bfsidata.json", "r", encoding="utf-8") as f:
    data = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("../models/bfsi_index.faiss")

# -------------------- POLICY QUERY DETECTOR --------------------

def is_policy_query(query: str) -> bool:
    keywords = [
        "regulatory", "framework", "compliance",
        "classification", "provisioning",
        "guidelines", "policy", "supervisory",
        "prudential", "reporting"
    ]
    q = query.lower()
    return any(k in q for k in keywords)

# -------------------- TIER 1 : DATASET MATCH --------------------

def tier1_response(user_query: str, threshold: float = 0.74):
    query_embedding = embedder.encode([user_query]).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, 1)
    similarity_score = float(scores[0][0])

    if similarity_score >= threshold:
        return {
            "tier": "dataset",
            "response": data[indices[0][0]]["output"],
            "score": similarity_score
        }

    return {
        "tier": "fallback",
        "response": None,
        "score": similarity_score
    }

# -------------------- MAIN RESPONSE ROUTER --------------------

# -------------------- MAIN RESPONSE ROUTER --------------------

def get_final_response(user_query: str):

    # ---- TIER 1: DATASET MATCH ----
    tier1 = tier1_response(user_query)
    print(f"[LOG] TIER 1 Score: {tier1['score']:.4f}")

    if tier1["tier"] == "dataset":
        return tier1["response"], "Tier 1 (Dataset)"

    # ---- TIER 3: RAG (Policy / Regulatory) ----
    if is_policy_query(user_query):
        context, rag_score = retrieve_context(user_query)
        print(f"[LOG] RAG Retrieval Score: {rag_score:.4f}")

        if context:
            # Clean RAG prompt: Only data, no role-play tags
            rag_prompt = f"Context: {context}\n\nQuestion: {user_query}"
            output = generate_response(rag_prompt)
            return output, "Tier 3 (RAG)"

    # ---- TIER 2: FINE-TUNED TINYLLAMA ----
    # Clean Tier 2 prompt: Just the query
    output = generate_response(user_query)
    return output, "Tier 2 (TinyLlama)"

# -------------------- CLI ENTRY --------------------

if __name__ == "__main__":
    query = input("\nCustomer Query: ")
    response, source = get_final_response(query)

    print("\n" + "=" * 40)
    print(f"FINAL SOURCE: {source}")
    print("RESPONSE:")
    print(response)
    print("=" * 40 + "\n")
