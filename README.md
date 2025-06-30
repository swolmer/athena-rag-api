# Athena Surgical RAG API

**Version:** 1.0  
**Author:** Sophie A. Wolmer  
**Last Updated:** June 2025  
**Domain Focus:**  Physician-Empowered, Ethical AI for Advanced Surgical Intelligence & Clinical Decision Support  
*(Built for Plastic and Reconstructive Surgery — extensible to all medical and surgical specialties)*  
**Language Model:** Hermes-2-Pro (Mistral-7B)

---

## Executive Summary

Athena Surgical RAG API is a next-generation, physician-centric Retrieval-Augmented Generation (RAG) platform engineered to return the power of AI to the hands of clinicians. Unlike black-box solutions, Athena enables surgeons and medical professionals to directly shape, train, and govern their AI assistant using their own trusted materials—eliminating the middleman and ensuring ethical, transparent, and specialty-specific intelligence.

Leveraging a state-of-the-art 7B parameter large language model, Athena seamlessly integrates domain-adapted document embeddings with a fast FAISS vector index. The result is an enterprise-grade system capable of delivering context-aware, evidence-based responses for surgical education, clinical workflow optimization, and real-time decision support—setting a new standard for trustworthy, physician-directed healthcare AI.

---

## Primary Use Cases

- Surgical education and case preparation
- Clinical documentation assistance/information retrieval
- Knowledge retrieval in EMR-integrated systems
- Intelligent interface for patient-facing surgical guidance (future modules)

---

## Architecture Overview

- **Language Model:** `NousResearch/Hermes-2-Pro-Mistral-7B`
- **Embedding Model:** `all-MiniLM-L6-v2` (via [sentence-transformers](https://www.sbert.net/))
- **Vector Indexing:** [FAISS](https://github.com/facebookresearch/faiss) (Flat L2 index, in-memory)
- **Document Types Supported:** `.pdf`, `.docx`, `.jpg`/`.png` (via OCR)
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)
- **Hardware Optimized For:** CUDA-enabled GPUs (e.g., RTX 4090)

---

## API Endpoints

### `POST /chat`

Accepts a clinical or surgical question and returns a structured model response using RAG over a curated corpus.

**Headers:**
- `x-api-key`: Required if `DEBUG=False`

**Request Body:**
```json
{
  "question": "How is a DIEP flap performed?",
  "slm_id": "plastic-surgery-rag-v1"
}
```

**Response:**
```json
{
  "question": "...",
  "answer": "...",
  "metadata": {
    "slm_id": "...",
    "overlap_score": 0.76,
    "hallucinated": false,
    "response_time_sec": 1.42
  }
}
```

### `GET /analytics/dashboard`

Returns usage metrics and hallucination analysis from the internal logging system. Useful for debugging, model evaluation, or tracking user query behavior.

**Output:**
```json
{
  "total_queries": 108,
  "hallucination_rate": 0.064,
  "avg_response_time_sec": 1.72,
  "avg_overlap_score": 0.81
}
```

### `GET /slm/list`

Returns available domain-specific language models (SLMs) available for routing.

---

## Environment Configuration

Ensure a .env file exists at the project root with the following entry:

```
RAG_API_KEY= kilment1234
```

---

## Setup Instructions

1. **Clone and Set Up Environment**
    ```bash
    git clone https://github.com/your-org/athena-rag-api.git
    cd athena-rag-api
    python -m venv .venv
    source .venv/bin/activate  # or .\.venv\Scripts\Activate on Windows
    pip install -r requirements.txt
    ```

2. **Build FAISS Index (if not already built)**
    ```bash
    python main_script.py --build_index
    ```

3. **Run the API Server**
    ```bash
    uvicorn api_server:app --reload
    ```

---

## Testing Example (via curl)

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -d "{\"question\": \"How is a DIEP flap performed?\", \"slm_id\": \"plastic-surgery-rag-v1\"}"
```

---

## Notes for Integration

- The RAG pipeline enforces hallucination filtering using token-level overlap scoring.
- Retrieval is re-ranked with cosine similarity after initial FAISS search to improve precision.
- Compatible with any front-end or clinical platform that can handle JSON-based HTTP requests.

---

## License and Use

This software and all associated materials are proprietary to Athena AI Platform and developed by Sophie A. Wolmer. The codebase is intended strictly for internal clinical use, research, and educational purposes within authorized organizations.

Unauthorized copying, redistribution, modification, or deployment—especially in regulated medical or commercial environments—is strictly prohibited without explicit written permission from the Athena AI Platform development team.

Users must comply with all applicable local, national, and international medical device regulations when deploying this software in clinical or commercial settings.

For licensing inquiries, commercial use, or partnership opportunities, please contact the Athena AI Platform development team directly.

