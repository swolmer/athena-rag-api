# api_server.py — Athena API (MVP)

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, time, json, pickle, numpy as np, faiss, torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# === Load from main_script.py ===
from main_script import (
    retrieve_context,
    generate_rag_answer_with_context,
    token_overlap_score
)

# ===========================
# CONFIGURATION
# ===========================

from dotenv import load_dotenv
load_dotenv()  # ✅ Load environment variables from .env at startup

LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Mistral-7B"

# ✅ API key is now securely loaded from the .env file
API_KEY = os.getenv("RAG_API_KEY", "")
if not API_KEY:
    print("⚠️ Warning: RAG_API_KEY is not set — API key check will be skipped unless DEBUG=True")

# ✅ Set DEBUG = False for production
DEBUG = False  # Set to True to disable API key check while testing locally

# ===========================
# LOAD MODEL + TOKENIZER
# ===========================

print(f"📦 Loading model and tokenizer: {LLM_MODEL_NAME}")

try:
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )

    print(f"✅ Model loaded on: {model.device}")
except Exception as e:
    print(f"❌ Failed to load model/tokenizer: {e}")
    raise e


# ===========================
# LOAD RAG INDEX
# ===========================
print("📡 Loading FAISS index and RAG context...")

try:
    rag_chunks = pickle.load(open("faiss_index/rag_chunks.pkl", "rb"))
    rag_embeddings = np.load("faiss_index/rag_embeddings.npy")
    faiss_index = faiss.read_index("faiss_index/faiss.index")

    # Register as globals for use in imported methods
    globals()["rag_chunks"] = rag_chunks
    globals()["rag_embeddings"] = rag_embeddings
    globals()["faiss_index"] = faiss_index
    globals()["rag_model"] = model

    print(f"✅ Loaded {len(rag_chunks)} chunks into FAISS index")

except Exception as e:
    print(f"❌ Failed to load RAG index or embeddings: {e}")
    raise e

# ===========================
# FASTAPI SETUP
# ===========================
app = FastAPI(title="Athena Surgical RAG API", version="1.0")

# ✅ CORS middleware — allow all origins for now (restrict in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Optional API key middleware — enforced only if DEBUG is False
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    protected_paths = ("/chat", "/analytics", "/slm")
    if not DEBUG and request.url.path.startswith(protected_paths):
        client_key = request.headers.get("x-api-key", "")
        if client_key != API_KEY:
            raise HTTPException(status_code=401, detail="Unauthorized: invalid API key")
    return await call_next(request)

# ===========================
# M1: /chat — Query RAG chatbot
# ===========================
@app.post("/chat")
async def chat(payload: dict):
    question = payload.get("question", "").strip()
    slm_id = payload.get("slm_id", "plastic-surgery-rag-v1")

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question' in request")

    try:
        start_time = time.time()

        # 🔍 Retrieve context
        context_chunks = retrieve_context(question)
        context_text = " ".join(context_chunks)

        # 🧠 Generate answer
        answer = generate_rag_answer_with_context(
            question, context_chunks, tokenizer, model
        )

        # 📊 Calculate hallucination risk
        overlap_score = token_overlap_score(answer, context_text)
        hallucinated = overlap_score < 0.35
        response_time = round(time.time() - start_time, 2)

        # 📝 Log query metadata
        log_query_metrics({
            "slm_id": slm_id,
            "question": question,
            "answer": answer,
            "context": context_chunks,
            "overlap_score": round(overlap_score, 3),
            "hallucinated": hallucinated,
            "response_time_sec": response_time,
            "timestamp": datetime.utcnow().isoformat()
        })

        # 📦 Return structured response
        return {
            "question": question,
            "answer": answer,
            "metadata": {
                "slm_id": slm_id,
                "overlap_score": round(overlap_score, 3),
                "hallucinated": hallucinated,
                "response_time_sec": response_time
            }
        }

    except Exception as e:
        logging.error(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ===========================
# M2: Logging + Analytics Dashboard
# ===========================

def log_query_metrics(entry: dict):
    """
    Append a single query log entry to logs/query_logs.jsonl.
    Each entry includes question, answer, overlap score, hallucination flag, and latency.
    """
    os.makedirs("logs", exist_ok=True)
    with open("logs/query_logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


@app.get("/analytics/dashboard")
async def analytics_dashboard():
    """
    Aggregate metrics from logged queries:
    - total queries
    - hallucination rate
    - average response time
    - average token overlap score
    """
    try:
        with open("logs/query_logs.jsonl", "r", encoding="utf-8") as f:
            logs = [json.loads(line) for line in f]
    except FileNotFoundError:
        return {"error": "No logs found."}

    if not logs:
        return {"error": "Log file is empty."}

    total = len(logs)
    hallucinations = sum(1 for x in logs if x.get("hallucinated"))
    total_time = sum(x.get("response_time_sec", 0.0) for x in logs)
    avg_overlap = np.mean([x.get("overlap_score", 0.0) for x in logs])

    return {
        "total_queries": total,
        "hallucination_rate": round(hallucinations / total, 3),
        "avg_response_time_sec": round(total_time / total, 2),
        "avg_overlap_score": round(avg_overlap, 3)
    }

# ===========================
# M3: List Available SLMs
# ===========================

@app.get("/slm/list")
async def list_slms():
    """
    Returns a list of all registered SLMs (structured language models).
    Used for client UI to filter by domain or application.
    """
    return [
        {
            "slm_id": "plastic-surgery-rag-v1",
            "status": "active",
            "usage": {
                "total_queries": 0,  # Future enhancement: count queries per SLM
                "avg_overlap": 0.0   # Future enhancement: calculate dynamically
            },
            "domain": "plastic surgery",
            "description": (
                "Structured RAG model for surgical reference questions, "
                "fine-tuned on reconstructive and aesthetic procedure materials."
            )
        }
    ]
