# main_api.py
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os

# Make sure these are imported from your main_script.py
from main_script import (
    retrieve_context,
    generate_rag_answer_with_context,
    tokenizer,
    rag_model,
)

# You can load your RAG_API_KEY here or from .env
API_KEY = os.getenv("RAG_API_KEY", "")

app = FastAPI(
    title="Athena Surgical RAG API",
    description="Retrieval-Augmented Generation Chatbot for surgical queries.",
    version="1.0",
)

class QueryRequest(BaseModel):
    question: str
    k: int | None = 3

@app.middleware("http")
async def check_api_key(request: Request, call_next):
    # Require API key in headers
    if "x-api-key" not in request.headers or request.headers["x-api-key"] != API_KEY:
        return JSONResponse(status_code=403, content={"error": "Forbidden. Invalid or missing API key."})
    return await call_next(request)

@app.post("/query")
async def query_rag(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Retrieve relevant context
    context_chunks = retrieve_context(request.question, k=request.k)
    if not context_chunks:
        return {"answer": "⚠️ No relevant context found to answer this question."}

    # Generate answer
    answer = generate_rag_answer_with_context(
        user_question=request.question,
        context_chunks=context_chunks,
        mistral_tokenizer=tokenizer,
        mistral_model=rag_model
    )

    return {
        "question": request.question,
        "answer": answer,
        "context_chunks": context_chunks
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}
