# BFSI Call Center AI Assistant

## Overview
This project is a local, compliance-focused AI assistant designed for BFSI (Banking, Financial Services, and Insurance) call center use cases.  
It prioritizes **dataset-grounded responses**, uses a **fine-tuned Small Language Model (SLM)**, and applies **RAG (Retrieval-Augmented Generation)** for policy-related queries.

The system is designed to be lightweight, safe, and production-oriented.

---

## Architecture



**Query Flow:**

1. **User Query** 2. → **Tier 1: Dataset Similarity (FAISS)** 3. → **Tier 2: Fine-Tuned TinyLlama (Local SLM)** 4. → **Tier 3: RAG (Policy / Regulatory Knowledge)** 5. → **Final Response**

---

## Key Features
- **Local Inference:** Using TinyLlama (1.1B) for privacy and speed.
- **Dataset Grounding:** Alpaca-style BFSI dataset (Instruction / Input / Output).
- **Semantic Search:** FAISS-based similarity matching.
- **Regulatory RAG:** Knowledge retrieval for policy-specific queries.
- **Strict Guardrails:** Programmatic enforcement to prevent hallucination.
- **Professional Tone:** Bullet-point, call-center-compliant responses.

---

## Tech Stack
- **Python**
- **PyTorch**
- **HuggingFace Transformers**
- **PEFT (LoRA)**
- **SentenceTransformers**
- **FAISS**

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt