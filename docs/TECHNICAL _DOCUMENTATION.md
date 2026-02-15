# BFSI Call Center AI Assistant - Technical Specification

## 1. System Overview
The **BFSI Call Center AI Assistant** is a lightweight, locally deployed AI system designed to handle common Banking, Financial Services, and Insurance (BFSI) customer queries in a safe, compliant, and deterministic manner. 

The system prioritizes pre-approved dataset responses and uses controlled language generation only when necessary. This design minimizes hallucinations, ensures regulatory compliance, and delivers consistent responses suitable for BFSI environments.

---

## 2. Design Principles
The system is built on the following core principles:

* **Safety First:** Dataset responses are always preferred over generation.
* **Determinism:** Avoid randomness in outputs to maintain consistency.
* **Local Execution:** No external API dependency, ensuring data privacy.
* **Regulatory Compliance:** No guessing of rates, policies, or customer data.
* **Cost Efficiency:** Lightweight models and FAISS-based retrieval for low-resource environments.

---

## 3. High-Level Architecture



### Architecture Flow
1.  **User Query**
2.  **Tier 1: Dataset Similarity Match (FAISS)** * *If no strong match...*
3.  **Tier 2: Fine-Tuned Small Language Model (TinyLlama)** * *If policy / regulatory intent detected...*
4.  **Tier 3: RAG (Knowledge Retrieval + Grounded Generation)**
5.  **Final Response**

Each tier is executed sequentially, ensuring that the safest and fastest option is always attempted first.

---

## 4. Component Breakdown

### 4.1 Dataset Layer (Tier 1 – Primary Response Layer)
**Purpose:** Provide pre-approved, compliant responses for frequently asked BFSI queries.

* **Dataset Format:** Alpaca-style JSON
    ```json
    {
      "instruction": "Explain EMI impact when interest rate increases",
      "input": "",
      "output": "An increase in interest rate may increase your EMI depending on the loan tenure and outstanding balance."
    }
    ```
* **Matching Mechanism:**
    * `SentenceTransformer (all-MiniLM-L6-v2)`
    * FAISS cosine similarity search.
    * Normalized embeddings with a configurable threshold.

### 4.2 Small Language Model (Tier 2 – Controlled Generation)
**Model:** `TinyLlama-1.1B-Chat` (Fine-Tuned with LoRA)

**Purpose:** Handle general BFSI queries not present in the dataset but not requiring regulatory documents.

* **Prompt Constraints:**
    * Bullet points only; no paragraphs.
    * No definitions unless explicitly asked.
    * Deterministic decoding (`do_sample=False`).
    * Explicit verification disclaimer if required.

### 4.3 RAG Layer (Tier 3 – Regulatory & Policy Queries)
**Purpose:** Answer complex policy or compliance-related queries using grounded knowledge.

* **Trigger Mechanism:** Deterministic keyword detection (e.g., *regulatory, compliance, provisioning, NPA, RBI*).
* **Knowledge Base:** RBI-style policy documents and loan restructuring guidelines.
* **Retrieval:** Section-based chunking with FAISS cosine similarity search.
* **Generation:** Context-injected prompt; model instructed to use **ONLY** retrieved content.

---

## 5. Tier Decision Logic

| Tier | Condition | Reason |
| :--- | :--- | :--- |
| **Tier 1** | High similarity match | Fastest & safest; zero hallucination risk. |
| **Tier 2** | No dataset match | Controlled flexibility for general queries. |
| **Tier 3** | Policy intent detected | Grounded compliance via official documents. |

---

## 6. Safety & Compliance Guardrails
The system explicitly prevents:
* ❌ Generating interest rates.
* ❌ Guessing internal bank policies.
* ❌ Exposing customer data.
* ❌ Out-of-domain responses.

**All outputs are concise, non-committal where required, and indicate when verification is needed.**

---

## 7. Scalability Considerations
* **Tier 1:** Scales horizontally via FAISS (memory-efficient).
* **Tier 2:** Optimized for CPU inference for moderate loads.
* **Tier 3:** Intentionally limited to reduce high-latency retrieval calls.

---

## 8. Version Control Strategy
| Asset | Strategy |
| :--- | :--- |
| **Dataset** | Semantic versioning (v1.0, v1.1) |
| **Fine-tuned Model** | Tagged LoRA checkpoints |
| **RAG Docs** | Date-based document versions |

---

## 9. Current Limitations
* Dataset coverage can be expanded further.
* RAG documents need deeper policy granularity.
* CLI demo (No Graphical User Interface yet).

---

## 10. Conclusion
This system demonstrates a production-oriented BFSI AI design that balances **Accuracy**, **Compliance**, and **Performance**. The tiered architecture ensures that generation is always the last resort, making the assistant suitable for real-world, highly regulated environments.