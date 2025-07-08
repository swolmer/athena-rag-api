# ===============================
# 🚀 1. IMPORTS
# ===============================
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import traceback

# ===============================
# 🔐 2. ENVIRONMENT SETUP
# ===============================
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("RAG_API_KEY", "")

# ===============================
# ⚙️ 3. FASTAPI APP INIT
# ===============================
app = FastAPI(
    title="Athena Surgical RAG API",
    description="Retrieval-Augmented Generation Chatbot for surgical queries.",
    version="1.0",
)

# ===============================
# 🌐 4. CORS MIDDLEWARE
# ===============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://athen.ai"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ===============================
# 🔒 5. API KEY MIDDLEWARE
# ===============================
@app.middleware("http")
async def check_api_key(request: Request, call_next):
    # ✅ Allow unauthenticated access to public endpoints
    public_paths = {"/docs", "/openapi.json", "/health", "/favicon.ico"}
    if request.url.path in public_paths:
        return await call_next(request)

    # 🔐 Enforce API key for all other requests
    if "x-api-key" not in request.headers or request.headers["x-api-key"] != API_KEY:
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden. Invalid or missing API key."}
        )

    return await call_next(request)

# ===============================
# 📦 6. REQUEST MODEL
# ===============================
class QueryRequest(BaseModel):
    question: str
    k: int | None = 3
    org_id: str
    collab: bool = False

# ===============================
# 🤖 7. MODEL IMPORTS + FAISS INIT
# ===============================
from main_script import (
    retrieve_context,
    generate_rag_answer_with_context,
    tokenizer,
    rag_model,
    load_rag_resources
)

# ✅ Preload FAISS index and chunks on API startup
load_rag_resources()
# ===============================
# 🔍 8. /QUERY ENDPOINT
# ===============================
@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        context_chunks = retrieve_context(
            query=request.question,
            k=request.k,
            org_id=request.org_id,
            collab=request.collab
        )
        if not context_chunks:
            return {
                "question": request.question,
                "org_id": request.org_id,
                "collab": request.collab,
                "answer": "⚠️ No relevant context found to answer this question.",
                "context_chunks": [],
                "hallucinated": True,
                "overlap_score": 0.0
            }

        answer = generate_rag_answer_with_context(
            user_question=request.question,
            context_chunks=context_chunks,
            mistral_tokenizer=tokenizer,
            mistral_model=rag_model
        )

        return {
            "question": request.question,
            "org_id": request.org_id,
            "collab": request.collab,
            "answer": answer,
            "context_chunks": context_chunks
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"error": f"Internal Server Error: {str(e)}"}
        )


# ===============================
# ❤️ 9. /HEALTH ENDPOINT
# ===============================
@app.get("/health")
async def health_check():
    return {"status": "ok"}
