# 📞 BFSI Call Center AI Assistant

A local, compliance-focused AI assistant designed for **Banking, Financial Services, and Insurance (BFSI)** call center queries.

Built using a fine-tuned **Small Language Model (SLM)** with dataset prioritization and **RAG-based policy retrieval**, ensuring safe and reliable responses.

---

## 🎯 Objective
To build a lightweight, fast, and regulatory-compliant AI system that:
* **Runs fully locally** to ensure data privacy and zero API costs.
* **Avoids hallucinated financial information** by using deterministic output control.
* **Prioritizes curated BFSI datasets** for approved answers.
* **Uses RAG only when required** for complex policy lookups.

---

## 🧠 System Architecture



**Query Routing Flow:**
1. **User Query Input**
2. **Tier 1: Alpaca Dataset Similarity (FAISS)**
   - *If similarity is high → Returns primary pre-approved response.*
3. **Tier 2: Fine-Tuned TinyLlama (Local SLM)**
   - *If no dataset match → Generates response via LoRA adapter.*
4. **Tier 3: RAG (FAISS + BFSI Policy Docs)**
   - *If policy intent detected → Grounds response in retrieved documents.*
5. **Final Response Generation**

---

## 🏗 Core Components

### 1️⃣ Dataset Layer (Tier 1)
* **Content:** 150+ Alpaca-formatted BFSI Q&A samples.
* **Tone:** Professional and compliant.
* **Role:** Primary source to ensure zero hallucination for standard queries.

### 2️⃣ Fine-Tuned SLM (Tier 2)
* **Base Model:** `TinyLlama-1.1B-Chat`
* **Adaptation:** Fine-tuned using **LoRA** (Low-Rank Adaptation).
* **Control:** Activated when the dataset does not contain a specific match.

### 3️⃣ RAG Layer (Tier 3)
* **Search:** FAISS-based semantic retrieval.
* **Knowledge:** Official BFSI regulatory and policy text documents.
* **Output:** Generates grounded, context-only answers to maintain compliance.

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

## 🛡 Guardrails & Compliance
* ❌ **No fabricated interest rates** or fake figures.
* ❌ **No assumptions** regarding bank policies.
* ❌ **No customer-specific data** exposure.
* ✅ **Deterministic responses** via greedy decoding logic.
* ✅ **Regulatory-safe language** enforced via post-processing filters.

---

## 📂 Project Structure

```text
BFSI-AI/
├── app/
│   ├── query_engine.py       # Main Response Router
│   ├── slm_engine.py         # TinyLlama Logic & Post-processing
│   ├── rag_engine.py         # RAG Logic
│   ├── fine_tune.py          # Training Script
│   └── build_rag_index.py    # Vector DB Setup
├── data/
│   └── bfsidata.json         # Tier 1 Q&A Dataset
├── rag_docs/
│   └── *.txt                 # Policy Documents
├── fine_tuned_model/         # LoRA Adapters
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md
├── requirements.txt
└── README.md
