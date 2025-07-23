# ============================
# 🚀 1. IMPORTS & GLOBAL STORAGE
# ============================

# --- Standard library ---
import os
import json
import logging
import pickle
import argparse
import re
import zipfile
import shutil
import traceback
import urllib.request
from pathlib import Path

# Set Hugging Face cache directory to a path with enough space on RunPod
os.environ["HF_HOME"] = "/workspace/huggingface_cache"

# --- Numerical / Data ---
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- Environment Variables ---
from dotenv import load_dotenv
load_dotenv()  # ✅ Load environment variables from .env file

HF_TOKEN = os.getenv("HF_TOKEN")
RAG_API_KEY = os.getenv("RAG_API_KEY")

if not RAG_API_KEY:
    print("⚠️ RAG_API_KEY is not set. Check your .env file.")

# --- Natural Language Processing ---
import nltk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.data.path.append(nltk_data_path)

# --- Transformers and Trainer ---
from transformers import (
    AutoConfig,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM,
    default_data_collator
)

# --- File Extraction Utilities ---
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document

# --- Web Scraping ---
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# --- Embedding & FAISS ---
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.metrics.pairwise import cosine_similarity

# --- Evaluation ---
from evaluate import load as load_metric

# --- FastAPI for web interface ---
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

def safe_sent_tokenize(text, lang='english'):
    """Safe sentence tokenization with fallback"""
    try:
        punkt_path = nltk.data.find(f'tokenizers/punkt/{lang}.pickle')
        with open(punkt_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.tokenize(text)
    except Exception as e:
        print(f"❌ NLTK sent_tokenize fallback used due to: {e}")
        return text.split('.')  # Fallback method

# ✅ CUDA Setup
print("🧠 Checking CUDA support:")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No CUDA-compatible GPU detected. Training will run on CPU.")

# ============================
# 🧠 2. GLOBAL ORG-SPECIFIC STORAGE WITH CLINICAL/NAVIGATION SEPARATION
# ============================

# These store model data per organization with clinical/navigation separation
ORG_FAISS_INDEXES = {}     # org_id → {"clinical": FAISS index, "navigation": FAISS index}
ORG_CHUNKS = {}            # org_id → {"clinical": list of chunks, "navigation": list of chunks}
ORG_EMBEDDINGS = {}        # org_id → {"clinical": np.ndarray, "navigation": np.ndarray}

# ✅ Tokenizer & Model will be shared
tokenizer = None
rag_model = None
embed_model = None

# ============================
# 🗂️ ORG-AWARE PATHS
# ============================

def get_org_paths(org_id):
    """Get all file paths for an organization"""
    base = os.path.join("org_data", org_id)
    return {
        "base": base,
        "clinical_training": os.path.join(base, "clinical_training"),
        "navigation_data": os.path.join(base, "extracted_content"),
        "html_pages": os.path.join(base, "html_pages"),
        "chunks_pkl": os.path.join(base, "chunks.pkl"),
        "embeddings_npy": os.path.join(base, "embeddings.npy"),
        "faiss_index": os.path.join(base, "index.faiss"),
        "training_data_dir": os.path.join(base, "training_data")
    }
tokenizer = None
rag_model = None
embed_model = None

# ============================
# 🛠️ 3. CONFIGURATION — COLLAB READY
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Parent folder for all org-specific data
ORG_DATA_ROOT = os.path.join(BASE_DIR, "org_data")  # e.g., ./org_data/emory/

# ✅ Shared model identifiers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Mistral-7B"

# ✅ Optional Hugging Face login
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ✅ Logging
logging.basicConfig(level=logging.INFO)

# ✅ CUDA Device Info
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Checking CUDA support:")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not available — using CPU")

# ============================
# 📁 4. ORG PATH HELPER
# ============================

def get_org_paths(org_id):
    """
    Returns all relevant file paths for a given organization.
    This isolates all indexes, embeddings, and model files by org_id.
    Now supports clinical vs navigation separation.
    Updated for RunPod deployment with correct clinical training paths.
    """
    base = os.path.join(ORG_DATA_ROOT, org_id)
    
    # For RunPod: Clinical materials are in root-level 'clinical' folder
    clinical_dir = os.path.join(BASE_DIR, "clinical")  # Root-level clinical folder
    
    return {
        "base": base,
        "clinical_training_dir": clinical_dir,  # Updated path for RunPod
        "navigation_training_dir": os.path.join(base, "navigation_training"),
        "clinical_faiss_index": os.path.join(base, "clinical_faiss_index.idx"),
        "navigation_faiss_index": os.path.join(base, "navigation_faiss_index.idx"),
        "clinical_chunks_pkl": os.path.join(base, "clinical_rag_chunks.pkl"),
        "navigation_chunks_pkl": os.path.join(base, "navigation_rag_chunks.pkl"),
        "clinical_embeddings_npy": os.path.join(base, "clinical_rag_embeddings.npy"),
        "navigation_embeddings_npy": os.path.join(base, "navigation_rag_embeddings.npy"),
        # Legacy paths for backward compatibility
        "training_data_dir": os.path.join(base, "training"),
        "faiss_index": os.path.join(base, "faiss_index.idx"),
        "chunks_pkl": os.path.join(base, "rag_chunks.pkl"),
        "embeddings_npy": os.path.join(base, "rag_embeddings.npy"),
        "model_dir": os.path.join(base, "model"),
        "csv_path": os.path.join(base, "Training_QA_Pairs.csv")  # optional
    }

# ============================
# 5. GLOBAL TOKENIZER & MODEL
# ============================

from transformers import AutoTokenizer, AutoModelForCausalLM

# Tokenizer (shared)
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Language Model (shared)
rag_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

__all__ = ["tokenizer", "rag_model"]

# ============================
# 6. GLOBAL EMBEDDING MODEL
# ============================

try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_model = embed_model.to(DEVICE)
    globals()["embed_model"] = embed_model
    logging.info(f"✅ Loaded embedding model '{EMBEDDING_MODEL_NAME}' on {DEVICE}")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model: {e}")
    embed_model = None

# ============================
# 7. UTILITIES — ENHANCED (HTML-ONLY)
# ============================

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def is_valid_chunk(text):
    text_lower = text.lower()
    skip_phrases = [
        "table of contents", "copyright", "terms and conditions",
        "accessibility statement", "website feedback", "http://", "https://"
    ]
    skip_starts = ["figure", "edition", "samir mardini", "flaps and"]

    if len(text.split()) <= 20:
        return False
    if any(phrase in text_lower for phrase in skip_phrases):
        return False
    if any(text_lower.strip().startswith(start) for start in skip_starts):
        return False

    return True

def safe_sent_tokenize(text, lang='english'):
    try:
        punkt_path = nltk.data.find(f'tokenizers/punkt/{lang}.pickle')
        with open(punkt_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.tokenize(text)
    except Exception as e:
        print(f"❌ NLTK sent_tokenize fallback used due to: {e}")
        return text.split('.')

def chunk_text_by_words(text, max_words=200, overlap=50, min_words=30):
    sentences = safe_sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sent in sentences:
        sent_words = sent.split()
        if len(current_chunk) + len(sent_words) > max_words:
            chunk_text = " ".join(current_chunk).strip()
            if len(current_chunk) >= min_words and is_valid_chunk(chunk_text):
                chunks.append(chunk_text)
            current_chunk = current_chunk[-overlap:] + sent_words
        else:
            current_chunk.extend(sent_words)

    if len(current_chunk) >= min_words:
        final_chunk = " ".join(current_chunk).strip()
        if is_valid_chunk(final_chunk):
            chunks.append(final_chunk)

    return chunks

# ============================
# 🎯 ENHANCED INTENT CLASSIFICATION FOR CLINICAL VS NAVIGATION
# ============================

def classify_question_intent(question):
    """
    Enhanced classification for clinical vs navigation questions.
    
    CLINICAL: Medical procedures, techniques, risks, recovery, complications
    NAVIGATION: Finding surgeons, costs, locations, appointments, general info
    """
    
    # Enhanced clinical keywords (medical content)
    clinical_keywords = [
        # Surgical procedures
        "surgery", "procedure", "surgical", "operation", "technique", "method",
        # Medical conditions
        "recovery", "healing", "risks", "complications", "anesthesia", 
        "post-operative", "pre-operative", "aftercare", "treatment",
        # Medical terminology
        "medical", "diagnosis", "symptoms", "patient", "tissue", "skin",
        "muscle", "bone", "reconstruction", "implant", "graft", "flap",
        # Specific procedures
        "rhinoplasty", "facelift", "liposuction", "augmentation", "reduction",
        "mastectomy", "reconstruction", "tummy tuck", "breast lift"
    ]
    
    # Enhanced navigation keywords (website/service info)
    navigation_keywords = [
        # Finding services
        "find", "locate", "search", "near me", "in my area", "directory",
        "surgeon", "doctor", "physician", "specialist", "clinic", "hospital",
        # Business/service info
        "cost", "price", "fee", "payment", "insurance", "financing", "affordable",
        "appointment", "consultation", "schedule", "book", "contact",
        "phone", "email", "address", "location", "hours", "available",
        # Organization info
        "about", "asps", "membership", "certification", "accreditation",
        "foundation", "news", "updates", "events", "education", "training",
        # Website navigation - ENHANCED
        "how to", "where can I", "who should I", "when should I",
        "website", "site", "plasticsurgery.org", "online", "web",
        "photos", "pictures", "gallery", "before and after", "results",
        "tool", "feature", "section", "page", "navigate", "access"
    ]
    
    # Weight certain phrases more heavily
    clinical_phrases = [
        "what is", "how is performed", "what are the risks", "recovery time",
        "surgical technique", "medical procedure", "complications of"
    ]
    
    navigation_phrases = [
        "find a surgeon", "cost of", "price of", "how much", "where to",
        "contact information", "make appointment", "schedule consultation",
        "where are the", "where can I see", "where exactly", "how do I use",
        "on the website", "on plasticsurgery.org", "before and after photos",
        "photo gallery", "find a tool", "use the tool", "navigate to",
        "where is the", "how to access", "where to find"
    ]
    
    question_lower = question.lower()
    
    # Score based on keywords
    clinical_score = sum(2 if keyword in question_lower else 0 for keyword in clinical_keywords)
    navigation_score = sum(2 if keyword in question_lower else 0 for keyword in navigation_keywords)
    
    # Score based on phrases (higher weight)
    clinical_score += sum(5 if phrase in question_lower else 0 for phrase in clinical_phrases)
    navigation_score += sum(5 if phrase in question_lower else 0 for phrase in navigation_phrases)
    
    # Additional heuristics
    if "?" in question and any(word in question_lower for word in ["how", "what", "when", "where", "why"]):
        if any(word in question_lower for word in ["procedure", "surgery", "technique", "recovery"]):
            clinical_score += 3
        elif any(word in question_lower for word in ["find", "cost", "price", "appointment", "photos", "website"]):
            navigation_score += 3
    
    # Strong website navigation indicators
    website_indicators = ["website", "plasticsurgery.org", "before and after", "photos", "gallery", "tool"]
    if any(indicator in question_lower for indicator in website_indicators):
        navigation_score += 8  # High weight for clear website questions
    
    # Debug output to help troubleshoot classification
    print(f"🎯 Intent scoring for: '{question}'")
    print(f"   Clinical score: {clinical_score}")
    print(f"   Navigation score: {navigation_score}")
    
    # Default to navigation for tie-breaking if website-related terms are present
    if any(indicator in question_lower for indicator in ["website", "photos", "where", "find", "how do i"]):
        if navigation_score >= clinical_score:
            return "navigation"
    
    # Otherwise default to clinical for medical tie-breaking
    if clinical_score > navigation_score:
        return "clinical"
    else:
        return "navigation"

def download_asps_subpages_as_html(html_dir="org_data/asps/html_pages"):
    """
    DEPRECATED - Use JSON knowledge bases instead.
    This function is kept for compatibility but not actively used.
    """
    print("⚠️ This function is deprecated - using JSON knowledge bases instead")
    return False
    """
    Downloads each ASPS cosmetic procedure page as raw HTML files.
    """
    os.makedirs(html_dir, exist_ok=True)
    root_url = "https://www.plasticsurgery.org/cosmetic-procedures"

    try:
        response = requests.get(root_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.select("a[href^='/cosmetic-procedures/']")

        sub_urls = {
            urljoin(root_url, a['href'])
            for a in links
            if a['href'].startswith("/cosmetic-procedures/")
            and not a['href'].endswith("/cosmetic-procedures")
        }

        for url in sorted(sub_urls):
            slug = url.split("/")[-1].strip()
            filepath = os.path.join(html_dir, f"{slug}.html")
            try:
                page_resp = requests.get(url)
                page_resp.raise_for_status()
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(page_resp.text)
                logging.info(f"✅ HTML saved: {filepath}")
            except Exception as e:
                logging.warning(f"⚠️ Failed to download or save {url} as HTML: {e}")

    except Exception as e:
        logging.error(f"❌ Failed to access ASPS page: {e}")

def extract_text_from_html(html_path):
    """
    Extracts meaningful text from an HTML file (mainly paragraphs inside <main> tag).
    """
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            paragraphs = soup.select("main p")
            text_blocks = [
                p.get_text(separator=" ", strip=True)
                for p in paragraphs
                if len(p.get_text(strip=True).split()) > 20
            ]
            return "\n\n".join(text_blocks)
    except Exception as e:
        logging.error(f"❌ HTML extraction failed for {html_path}: {e}")
        return ""

def extract_all_text_from_asps_html(html_dir="org_data/asps/html_pages"):
    """
    Goes through all ASPS HTML files and extracts combined text.
    """
    if not os.path.exists(html_dir):
        raise ValueError(f"❌ HTML directory not found: {html_dir}")

    all_text = ""
    for file in os.listdir(html_dir):
        if not file.endswith(".html"):
            continue
        full_path = os.path.join(html_dir, file)
        html_text = extract_text_from_html(full_path)
        if html_text.strip():
            all_text += "\n\n" + html_text
        else:
            logging.warning(f"⚠️ Empty or unreadable HTML: {file}")

    return all_text

# ============================
# 8. DATASET CLASS
# ============================

class MistralQADataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=512, debug=False):
        """
        Loads a dataset from a JSONL file and prepares it for causal LM fine-tuning.
        Each line in the file should contain: {"instruction": ..., "output": ...}
        """
        self.samples = []
        self.debug = debug

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                try:
                    item = json.loads(line)

                    # Format prompt + response
                    prompt = f"### Instruction:\n{item['instruction']}\n\n### Response:\n"
                    answer = item["output"]
                    full_text = prompt + answer

                    # Tokenize full example
                    tokenized = tokenizer(
                        full_text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                        return_tensors="pt"
                    )

                    input_ids = tokenized["input_ids"].squeeze(0)
                    attention_mask = tokenized["attention_mask"].squeeze(0)

                    # Labels are the same as input_ids, but padding gets -100
                    labels = input_ids.clone()
                    labels[labels == tokenizer.pad_token_id] = -100

                    if self.debug and i < 3:
                        print("input_ids:", input_ids)
                        print("labels:", labels)
                        print("Any label != -100?", (labels != -100).any().item())

                    # Store example
                    self.samples.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels
                    })

                    # Debug print
                    if self.debug and i < 3:
                        print(f"\n🔍 DEBUG SAMPLE {i}")
                        print("Instruction:", item["instruction"])
                        print("Output:", item["output"])
                        print("input_ids:", input_ids.tolist())
                        print("labels:", labels.tolist())
                        print("Unique labels:", torch.unique(labels).tolist())
                        print("Token length:", input_ids.shape[0])
                        print("-" * 50)

                except Exception as e:
                    logging.warning(f"❌ Skipping malformed line {i}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "input_ids": sample["input_ids"],
            "attention_mask": sample["attention_mask"],
            "labels": sample["labels"]
        }

# ============================
# 9. TRAINING
# ============================

from transformers import default_data_collator

# ✅ Custom Trainer to prevent re-moving model to device
class CustomTrainer(Trainer):
    def _move_model_to_device(self, model, device):
        return model  # Avoid double GPU mapping

def fine_tune_with_trainer(
    train_dataset,
    eval_dataset,
    model,
    tokenizer,
    output_dir,
    debug=False
):
    """
    Fine-tunes a language model using Hugging Face's Trainer API.
    Includes gradient check, loss verification, and logging.
    """
    assert isinstance(output_dir, str), f"`output_dir` must be a string, got {type(output_dir)}"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        bf16=False,
        torch_compile=False,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        remove_unused_columns=False,
        skip_memory_metrics=True
    )

    if debug:
        print("🔍 Checking sample input/output tensors...")
        for i in range(min(5, len(train_dataset))):
            sample = train_dataset[i]
            print(f"📦 Sample {i}")
            print("   input_ids[:10]:", sample["input_ids"][:10])
            print("   labels[:10]:", sample["labels"][:10])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sample_input = {k: v.unsqueeze(0).to(device) for k, v in train_dataset[0].items()}
        model.train()
        outputs = model(**sample_input)

        print("🚨 Forward output keys:", outputs.keys() if hasattr(outputs, "keys") else outputs)
        print("🚨 Has loss:", hasattr(outputs, "loss"))
        if hasattr(outputs, "loss") and outputs.loss is not None:
            print("🚨 Loss value:", outputs.loss.item())
            if torch.isnan(outputs.loss):
                raise ValueError("❌ NaN loss encountered — check your dataset formatting.")
        else:
            raise ValueError("❌ Model did not return a loss — likely a label formatting issue.")

        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"✅ Trainable parameters: {len(trainable)}")

    # ✅ Launch Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    print("🚀 Starting training...")
    trainer.train()

    # ✅ Save final model + tokenizer
    print("💾 Saving model and tokenizer to:", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# ============================
# 10. RETRIEVAL FUNCTION — FIXED
# ============================

def retrieve_context(query, k=3, initial_k=10, org_id=None, intent=None, collab=False):
    """
    Retrieves k most relevant chunks based on intent (clinical or navigation).
    If collab=True, you could merge chunks from multiple orgs (not implemented here).
    If collab=False, retrieves from specified org_id and intent type.
    
    Parameters:
    - query: Search query
    - k: Number of final chunks to return
    - initial_k: Number of initial candidates to consider
    - org_id: Organization identifier
    - intent: Either "clinical" or "navigation" (auto-detected if None)
    - collab: Whether to use collaborative search (not implemented)
    """
    if not query or not hasattr(embed_model, 'encode'):
        raise ValueError("❌ Query is empty or embed_model not initialized.")

    if collab:
        raise NotImplementedError("Collaborative retrieval is not implemented yet.")
    if not org_id:
        raise ValueError("❌ 'org_id' must be provided if collab=False.")
    
    # Auto-detect intent if not provided
    if intent is None:
        intent = classify_question_intent(query)
        print(f"🎯 Auto-detected intent: {intent}")
    
    if intent not in ["clinical", "navigation"]:
        raise ValueError("❌ 'intent' must be either 'clinical' or 'navigation'.")

    # Load per-org data for specific intent
    org_indexes = ORG_FAISS_INDEXES.get(org_id, {})
    org_chunks = ORG_CHUNKS.get(org_id, {})
    org_embeddings = ORG_EMBEDDINGS.get(org_id, {})
    
    faiss_index = org_indexes.get(intent)
    rag_chunks = org_chunks.get(intent)
    rag_embeddings = org_embeddings.get(intent)

    if (
        faiss_index is None or
        rag_chunks is None or len(rag_chunks) == 0 or
        rag_embeddings is None or len(rag_embeddings) == 0
    ):
        print(f"⚠️ No {intent} data available for org '{org_id}'. Available: {list(org_indexes.keys())}")
        # Fallback to the other intent if available
        fallback_intent = "navigation" if intent == "clinical" else "clinical"
        fallback_index = org_indexes.get(fallback_intent)
        fallback_chunks = org_chunks.get(fallback_intent)
        fallback_embeddings = org_embeddings.get(fallback_intent)
        
        if fallback_index is not None and fallback_chunks and fallback_embeddings is not None:
            print(f"🔄 Falling back to {fallback_intent} data")
            faiss_index = fallback_index
            rag_chunks = fallback_chunks
            rag_embeddings = fallback_embeddings
        else:
            raise ValueError(f"❌ No FAISS data available for org_id '{org_id}' (tried {intent} and {fallback_intent})")

    try:
        # Encode query
        query_embedding = embed_model.encode(
            query,
            convert_to_tensor=True
        ).cpu().numpy().reshape(1, -1)

        # Search in FAISS index
        D, I = faiss_index.search(query_embedding, initial_k)

        # Retrieve candidate chunks and embeddings
        candidate_chunks = [rag_chunks[i] for i in I[0]]
        candidate_embeddings = [rag_embeddings[i] for i in I[0]]

        # Compute cosine similarity
        scores = cosine_similarity(query_embedding, candidate_embeddings)[0]

        # Rank candidates by similarity
        ranked = sorted(
            zip(candidate_chunks, scores),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top-k chunks
        return [chunk for chunk, _ in ranked[:k]]

    except Exception as e:
        logging.error(f"❌ Failed to retrieve context for intent '{intent}': {e}")
        return []
# ============================
# 11. RAG GENERATION — FIXED
# ============================

def generate_rag_answer_with_context(user_question, context_chunks, mistral_tokenizer, mistral_model, intent="clinical"):
    if not context_chunks:
        if intent == "navigation":
            return ("I don't have specific information about this ASPS website navigation question in my current knowledge base. "
                   "Rather than provide potentially outdated guidance, I recommend:\n\n"
                   "📍 Visit plasticsurgery.org directly for current information\n"
                   "🔍 Use their site search function for specific topics\n"
                   "📞 Contact ASPS support at (847) 228-9900 for personalized assistance\n\n"
                   "This ensures you get the most accurate and up-to-date information about their website features and resources.")
        else:
            return ("I prioritize your safety and health by not providing medical information I cannot verify from my training materials. "
                   "Rather than risk giving you incorrect clinical guidance, I strongly recommend:\n\n"
                   "🩺 Consult a board-certified plastic surgeon - They can provide personalized, evidence-based advice\n"
                   "📚 Review peer-reviewed literature - Look for recent studies on your specific concern\n"
                   "🏥 Speak with your healthcare provider - They know your medical history and current health status\n"
                   "📞 Contact ASPS at (847) 228-9900 for surgeon referrals in your area\n\n"
                   "Your health and safety are paramount - professional medical consultation is always the safest approach for clinical questions.")

    context = "\n\n".join(f"- {chunk.strip()}" for chunk in context_chunks)
    
    # Different formats based on question intent
    if intent == "navigation":
        # Navigation format for website/service questions - actionable and specific
        prompt = (
            "You are a helpful assistant providing specific, actionable guidance about ASPS resources and services.\n"
            "Use only the CONTEXT below to answer the QUESTION with direct, specific instructions.\n"
            "Be actionable - tell users exactly what to do, not just where to go.\n"
            "Avoid generic phrases like 'on the website' or 'visit the site'.\n"
            "Format your response as:\n\n"
            "📍 Direct Answer: (1-2 specific, actionable sentences)\n"
            "🔗 How to Find It: (step-by-step directions or direct URL if available)\n"
            "💡 Additional Help: (specific next steps or related resources)\n\n"
            f"### CONTEXT:\n{context}\n\n"
            f"### QUESTION:\n{user_question}\n\n"
            f"### ANSWER:\n"
        )
    else:
        # Clinical format for medical/surgical questions
        prompt = (
            "You are a surgical expert writing answers for a clinical reference guide.\n"
            "Use only the CONTEXT below to answer the QUESTION in this clinical format:\n\n"
            "✅ Summary: (1 sentence)\n"
            "🧠 Anatomy & Physiology:\n"
            "🔧 Procedure or Technique:\n"
            "⚠️ Pitfalls & Pearls:\n\n"
            f"### CONTEXT:\n{context}\n\n"
            f"### QUESTION:\n{user_question}\n\n"
            f"### ANSWER:\n"
        )

    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3072).to(mistral_model.device)

    with torch.no_grad():
        outputs = mistral_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=512,     # increased to 512 for fuller answers without truncation
            do_sample=False,        # deterministic, faster
            temperature=0.1,        # low temperature for more focused responses
            repetition_penalty=1.2, # prevent repetitive text
            eos_token_id=mistral_tokenizer.eos_token_id,
            pad_token_id=mistral_tokenizer.pad_token_id,
            early_stopping=False    # disable early stopping to prevent truncation
        )

    decoded = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract portion after '### ANSWER:'
    if "### ANSWER:" in decoded:
        answer = decoded.split("### ANSWER:")[-1].strip()
    else:
        answer = decoded.strip()
    # Clean up repeated characters and invalid sequences
    import re
    
    # Remove the specific pattern of repeated numbers with spaces (like "1 3 2 3 3 3 4 3...")
    answer = re.sub(r'\b(\d+\s+){4,}.*$', '', answer)    # Remove sequences of 4+ numbers with spaces at end
    answer = re.sub(r'\s+(\d+\s+){3,}.*$', '', answer)   # Remove sequences of 3+ spaced numbers anywhere
    
    # Remove excessive repeated characters (like "9 9 9 9 9...")
    answer = re.sub(r'\b(\d)\s+\1(\s+\1)+', '', answer)  # Remove repeated digits with spaces
    answer = re.sub(r'(\w)\1{3,}', r'\1', answer)        # Remove excessive character repetition
    
    # Remove trailing number sequences and artifacts
    answer = re.sub(r'\s+\d+(\s+\d+)*\s*$', '', answer) # Remove trailing number sequences
    answer = re.sub(r'\s+[0-9\s]+$', '', answer)         # Remove any trailing numbers and spaces
    
    # Clean up whitespace and common artifacts
    answer = re.sub(r'\s+', ' ', answer)                 # Normalize whitespace
    answer = re.sub(r'\b(the the|and and|of of|in in)\b', r'\1'.split()[0], answer)  # Remove repeated words
    answer = re.sub(r'[^\w\s\.,!?:;()-]', '', answer)    # Remove invalid characters
    
    # Safe truncation after punctuation marks (., !, ?) - keep up to 12 sentences for fuller answers
    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > 12:
        answer = " ".join(sentences[:12]).strip()
    else:
        answer = answer.strip()

    # Final cleanup after sentence truncation - remove any remaining number artifacts
    answer = re.sub(r'\s+\d+(\s+\d+)*\s*$', '', answer)  # Remove any trailing number sequences
    answer = re.sub(r'\s+[0-9\s]+$', '', answer)          # Remove trailing numbers and spaces
    
    # Additional safety check for the specific pattern you encountered
    answer = re.sub(r'\.\s*\d+(\s+\d+)+.*$', '.', answer)  # Remove anything after a period followed by number sequences
    answer = re.sub(r'\s+\d+\s+\d+\s+\d+.*$', '', answer)  # Remove sequences of 3+ spaced numbers at end
    
    answer = answer.strip()                               # Clean up whitespace

    # Ensure answer ends with a period
    if not answer.endswith(('.', '!', '?')):
        answer += "."

    # Hallucination filter: check token overlap with context
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    overlap = answer_tokens & context_tokens
    overlap_score = len(overlap) / max(1, len(answer_tokens))

    if overlap_score < 0.25:
        logging.warning("⚠️ Low token overlap — likely hallucination.")
        if intent == "navigation":
            return ("I don't have enough reliable information to provide accurate guidance about this specific website navigation question. "
                   "Rather than potentially mislead you, I recommend getting current information directly from:\n\n"
                   "📍 plasticsurgery.org - Visit the official site for the most up-to-date features\n"
                   "🔍 Site search - Use their search function for specific topics\n"
                   "📞 ASPS support - Call (847) 228-9900 for personalized website assistance\n"
                   "💬 Live chat - Check if they offer live support for navigation questions\n\n"
                   "This ensures you get accurate, current information about their website resources and tools.")
        else:
            return ("I cannot provide confident medical information for this specific clinical question, as I prioritize accuracy and your safety above all else. "
                   "Rather than risk giving you potentially incorrect medical guidance that could affect your health decisions, I strongly recommend:\n\n"
                   "🩺 Board-certified plastic surgeon consultation - Get personalized, professional medical advice\n"
                   "📚 Peer-reviewed medical literature - Research current, evidence-based studies on your topic\n"
                   "🏥 Your healthcare provider - They understand your medical history and current health status\n"
                   "📞 ASPS surgeon referral - Call (847) 228-9900 to find qualified specialists in your area\n"
                   "🌐 ASPS patient education resources - Visit plasticsurgery.org for verified patient information\n\n"
                   "Your health and safety are my top priorities - professional medical consultation is always the most reliable path for clinical guidance.")

    return answer


# ============================
# 12. EVALUATION FUNCTION — FIXED
# ============================

def token_overlap_score(answer: str, context: str) -> float:
    """
    Calculates token-level overlap between answer and context for hallucination risk estimation.
    """
    import re
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    overlap = answer_tokens & context_tokens
    return len(overlap) / max(1, len(answer_tokens))


def evaluate_on_examples(model, tokenizer, sample_questions, save_path="eval_outputs.json", k=3, org_id=None):
    """
    Evaluates the RAG chatbot on a list of questions using retrieved context and overlap scoring.
    Saves structured results to a JSON file.

    Parameters:
    - model: Hugging Face language model (Mistral or similar)
    - tokenizer: Corresponding tokenizer
    - sample_questions (list of str): Questions to evaluate
    - save_path (str): Where to store the evaluation output
    - k (int): Number of chunks to retrieve
    - org_id (str): Organization ID for selecting the correct FAISS index and training data
    """
    global rag_model, faiss_index, rag_chunks, rag_embeddings  # required globals

    outputs = []

    for idx, question in enumerate(sample_questions, 1):
        print(f"\n🔹 Question {idx}/{len(sample_questions)}: {question}")

        try:
            # Step 1: Classify question intent
            intent = classify_question_intent(question)
            
            # Step 2: Retrieve top-k chunks
            context_chunks = retrieve_context(query=question, k=k, org_id=org_id, intent=intent)
            context_combined = " ".join(context_chunks)

            # Step 3: Generate answer from model
            answer = generate_rag_answer_with_context(
                user_question=question,
                context_chunks=context_chunks,
                mistral_tokenizer=tokenizer,
                mistral_model=model,
                intent=intent
            )

            # Step 3: Compute token overlap
            overlap_score = token_overlap_score(answer, context_combined)
            hallucinated = overlap_score < 0.25

            if hallucinated:
                logging.warning(f"⚠️ Token Overlap = {overlap_score:.2f} — potential hallucination.")
                if intent == "navigation":
                    answer = ("I don't have enough reliable information to provide accurate website navigation guidance for this question. "
                             "Please visit plasticsurgery.org directly or contact ASPS support at (847) 228-9900 for current, accurate information.")
                else:
                    answer = ("I prioritize your safety by not providing potentially inaccurate medical information. "
                             "Please consult with a board-certified plastic surgeon or your healthcare provider for reliable clinical guidance. "
                             "You can find qualified surgeons through ASPS at (847) 228-9900 or plasticsurgery.org.")

            print("✅ Answer:", answer)

            outputs.append({
                "question": question,
                "context_chunks": context_chunks,
                "answer": answer,
                "overlap_score": round(overlap_score, 3),
                "hallucination_flag": hallucinated
            })

        except Exception as e:
            logging.error(f"❌ Error generating answer for question {idx}: {e}")
            outputs.append({
                "question": question,
                "context_chunks": [],
                "answer": f"Error: {e}",
                "overlap_score": None,
                "hallucination_flag": True
            })

    # Save evaluation results
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(outputs, f, indent=2)
        print(f"📁 Evaluation results saved to: {save_path}")
    except Exception as e:
        logging.error(f"❌ Failed to save evaluation results: {e}")

# ============================
# � 13. LOAD FAISS INTO MEMORY
# ============================

def load_faiss_into_memory(org_id):
    """
    Loads the org-specific index/chunks/embeddings into the global lookup dicts.
    This makes RAG work immediately after ASPS data loading.
    """
    paths = get_org_paths(org_id)
    try:
        with open(paths["chunks_pkl"], "rb") as f:
            ORG_CHUNKS[org_id] = pickle.load(f)
        ORG_EMBEDDINGS[org_id] = np.load(paths["embeddings_npy"])
        ORG_FAISS_INDEXES[org_id] = faiss.read_index(paths["faiss_index"])
        logging.info(f"✅ FAISS memory loaded for org '{org_id}'")
    except Exception as e:
        logging.error(f"❌ Failed to load FAISS into memory for org '{org_id}': {e}")
        raise

# ============================
# 📥 14. Build FAISS Index
# ============================

# ============================
# ⚡ PERFORMANCE OPTIMIZATIONS FOR RUNPOD
# ============================

def optimize_for_speed():
    """
    Applies performance optimizations for RunPod deployment.
    """
    import torch
    
    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Set optimal batch sizes based on GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if "4090" in gpu_name or "a100" in gpu_name:
            return {"batch_size": 32, "max_workers": 8}
        elif "3080" in gpu_name or "a6000" in gpu_name:
            return {"batch_size": 16, "max_workers": 4}
    
    return {"batch_size": 8, "max_workers": 2}

def parallel_pdf_processing(pdf_files, max_workers=4):
    """
    Process multiple PDFs in parallel for faster extraction.
    """
    import concurrent.futures
    from functools import partial
    
    print(f"🚀 Processing {len(pdf_files)} PDFs with {max_workers} workers...")
    
    def process_single_pdf(pdf_path):
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            if raw_text:
                chunks = chunk_text_by_words(raw_text)
                return [c for c in chunks if is_valid_chunk(c)]
        except Exception as e:
            print(f"❌ Failed to process {pdf_path}: {e}")
        return []
    
    all_chunks = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pdf = {executor.submit(process_single_pdf, pdf): pdf for pdf in pdf_files}
        
        for future in concurrent.futures.as_completed(future_to_pdf):
            pdf = future_to_pdf[future]
            try:
                chunks = future.result()
                all_chunks.extend(chunks)
                print(f"✅ Processed {os.path.basename(pdf)}: {len(chunks)} chunks")
            except Exception as e:
                print(f"❌ Error processing {pdf}: {e}")
    
    return all_chunks

# ============================
# 🔍 RUNPOD DEPLOYMENT VERIFICATION
# ============================

def verify_clinical_training_setup():
    """
    Verifies all clinical training materials are accessible for RunPod deployment.
    
    For RunPod with GitHub token:
    - First checks for GitHub repository access with required files
    - Then checks for local clinical training directories
    - Returns a summary of found training materials
    """
    import urllib.request
    
    print("🔍 Verifying clinical training setup for RunPod deployment...")
    
    # Define all clinical training directories that should be in GitHub repo (3 blue folders only)
    clinical_dirs_github = [
        "clinical",         # Blue folder 1
        "op notes",         # Blue folder 2  
        "textbook notes"    # Blue folder 3
    ]
    
    # Also check local paths (fallback for development) - 3 blue folders only
    clinical_dirs_local = [
        ("clinical", os.path.join(BASE_DIR, "clinical")),           # Blue folder 1
        ("op notes", os.path.join(BASE_DIR, "op notes")),           # Blue folder 2
        ("textbook notes", os.path.join(BASE_DIR, "textbook notes")) # Blue folder 3
    ]
    
    # Required GitHub JSON files
    github_json_files = [
        "navigation_training_data.json",
        "nav1.json",
        "nav2.json"
    ]
    
    summary = {
        "total_pdf_files": 0,
        "total_docx_files": 0,
        "directories_found": [],
        "directories_missing": [],
        "github_files_available": [],
        "github_files_missing": [],
        "total_size_mb": 0,
        "deployment_ready": False
    }
    
    print("📋 Checking GitHub repository access...")
    
    # Check GitHub JSON files accessibility
    github_repo = "swolmer/athena-rag-api"  # Your repository
    github_branch = "asps_demo"  # Use the correct branch
    github_base_url = f"https://raw.githubusercontent.com/{github_repo}/{github_branch}"
    
    for filename in github_json_files:
        try:
            url = f"{github_base_url}/{filename}"
            response = urllib.request.urlopen(url)
            if response.getcode() == 200:
                file_size = len(response.read()) / (1024 * 1024)  # MB
                summary["github_files_available"].append(f"{filename} ({file_size:.1f}MB)")
                print(f"✅ GitHub: {filename} accessible ({file_size:.1f}MB)")
            else:
                summary["github_files_missing"].append(filename)
                print(f"❌ GitHub: {filename} not accessible")
        except Exception as e:
            summary["github_files_missing"].append(filename)
            print(f"❌ GitHub: {filename} error - {e}")
    
    print("📁 Checking local clinical training directories...")
    
    # Check local directories (for development/testing)
    for dir_name, dir_path in clinical_dirs_local:
        if os.path.exists(dir_path):
            print(f"✅ Local: {dir_name} at {dir_path}")
            summary["directories_found"].append(dir_name)
            
            # Count files and calculate size
            for root, _, files in os.walk(dir_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                    summary["total_size_mb"] += file_size
                    
                    if file.endswith(".pdf"):
                        summary["total_pdf_files"] += 1
                    elif file.endswith(".docx"):
                        summary["total_docx_files"] += 1
        else:
            print(f"⚠️ Local: {dir_name} not found at {dir_path}")
            summary["directories_missing"].append(dir_name)
    
    print("\n📊 TRAINING MATERIALS VERIFICATION SUMMARY:")
    print(f"   🌐 GitHub JSON files accessible: {len(summary['github_files_available'])}")
    for file_info in summary["github_files_available"]:
        print(f"      - {file_info}")
    
    if summary["github_files_missing"]:
        print(f"   ❌ GitHub files missing: {summary['github_files_missing']}")
    
    print(f"   📁 Local directories found: {len(summary['directories_found'])}")
    if summary["directories_found"]:
        print(f"   📄 Local PDF files: {summary['total_pdf_files']}")
        print(f"   📝 Local DOCX files: {summary['total_docx_files']}")
        print(f"   💾 Local total size: {summary['total_size_mb']:.1f} MB")
    
    if summary["directories_missing"]:
        print(f"   ⚠️ Local directories missing: {summary['directories_missing']}")
    
    # Determine deployment readiness
    github_json_ready = len(summary["github_files_available"]) >= 3  # nav1, nav2, navigation_training_data
    clinical_data_ready = len(summary["directories_found"]) > 0  # Has some clinical directories
    
    summary["deployment_ready"] = github_json_ready and clinical_data_ready
    
    if github_json_ready and clinical_data_ready:
        print(f"\n🚀 RunPod Status: ✅ Ready for deployment!")
        print(f"   📋 GitHub JSON knowledge bases: ✅ Available")
        print(f"   📚 Clinical training materials: ✅ Available")
    elif github_json_ready:
        print(f"\n🟡 RunPod Status: Partial - GitHub files ready, but no clinical training directories found")
        print(f"   📋 GitHub JSON knowledge bases: ✅ Available") 
        print(f"   📚 Clinical training materials: ❌ Missing locally (should be in GitHub)")
    else:
        print(f"\n❌ RunPod Status: Not ready - Missing required files")
        print(f"   📋 GitHub JSON knowledge bases: {'✅' if github_json_ready else '❌'}")
        print(f"   📚 Clinical training materials: {'✅' if clinical_data_ready else '❌'}")
    
    return summary

def print_runpod_deployment_checklist():
    """
    Prints a comprehensive checklist for RunPod deployment with GitHub integration.
    """
    print("\n" + "="*60)
    print("🚀 RUNPOD DEPLOYMENT CHECKLIST")
    print("="*60)
    
    print("\n📋 REQUIRED FILES IN YOUR GITHUB REPOSITORY:")
    print("   Repository: swolmer/athena-rag-api")
    print("   Branch: main")
    print("")
    
    print("📄 JSON Knowledge Base Files (Root Directory) - ALL GO TO NAVIGATION:")
    print("   ✅ navigation_training_data.json   - Original navigation training data")
    print("   ✅ nav1.json                       - Goes to NAVIGATION index")
    print("   ✅ nav2.json                       - Goes to NAVIGATION index")
    print("   ✅ comprehensive_split_*.json      - All splits go to NAVIGATION index")
    print("   📦 ultimate_asps_knowledge_base.json - Backup file for NAVIGATION")
    print("")
    
    print("📁 Clinical Training Directories (3 Blue Folders Only) - GO TO CLINICAL:")
    print("   ✅ clinical/                       - Clinical materials folder (blue)")
    print("   ✅ op notes/                       - Operative notes folder (blue)")
    print("   ✅ textbook notes/                 - Textbook notes folder (blue)")
    print("")
    
    print("🔧 RUNPOD DEPLOYMENT PROCESS:")
    print("   1️⃣ Provide GitHub token in RunPod environment")
    print("   2️⃣ Clone repository: git clone https://github.com/swolmer/athena-rag-api.git")
    print("   3️⃣ Run: python demo_asps.py")
    print("   4️⃣ System will automatically:")
    print("      - Download ALL JSON files (nav1.json, nav2.json, navigation_training_data.json, comprehensive_split_*.json)")
    print("      - Load clinical training directories from 3 blue folders only")
    print("      - Build dual FAISS indexes (clinical from folders + navigation from JSON)")
    print("      - Start training/inference")
    print("")
    
    print("💡 INDEX MAPPING:")
    print("   🧭 NAVIGATION INDEX uses:")
    print("      - ALL JSON files (nav1.json, nav2.json, navigation_training_data.json)")
    print("      - comprehensive_split_*.json files")
    print("      - ultimate_asps_knowledge_base.json")
    print("")
    print("   📚 CLINICAL INDEX uses:")
    print("      - clinical/ folder (blue folder)")
    print("      - op notes/ folder (blue folder)")
    print("      - textbook notes/ folder (blue folder)")
    print("")
    
    print("🎯 FINAL VERIFICATION:")
    print("   Run verify_clinical_training_setup() to confirm all files accessible")
    print("="*60)

# ============================
# 🏗️ BUILD CLINICAL & NAVIGATION FAISS INDEXES SEPARATELY
# ============================

def build_clinical_navigation_indexes(org_id="asps"):
    """
    Builds separate FAISS indexes for clinical and navigation content.
    
    NAVIGATION: Uses ALL JSON files (nav1.json, nav2.json, navigation_training_data.json, comprehensive_split_*.json)
    CLINICAL: Uses ONLY 3 blue folders (clinical/, op notes/, textbook notes/)
    """
    print(f"🏗️ Building clinical and navigation indexes for '{org_id}'...")
    paths = get_org_paths(org_id)
    
    # Create base directory
    os.makedirs(paths["base"], exist_ok=True)
    
    # Initialize storage for both types
    clinical_chunks = []
    navigation_chunks = []
    
    # ============================
    # 📚 STEP 1: BUILD CLINICAL INDEX (Original Training Materials)
    # ============================
    print("📚 Processing CLINICAL training materials...")
    
    # Look for clinical training materials - use the correct path from get_org_paths
    clinical_training_dir = paths["clinical_training_dir"]
    
    # Define all possible clinical training directories based on user's structure
    potential_clinical_dirs = [
        clinical_training_dir,  # Root-level clinical folder
        os.path.join(BASE_DIR, "Training Data Op"),
        os.path.join(BASE_DIR, "Training Data Textbooks"), 
        os.path.join(BASE_DIR, "Validate"),
        os.path.join(BASE_DIR, "op notes"),
        os.path.join(BASE_DIR, "textbook notes")
    ]
    
    found_clinical_data = False
    
    for potential_dir in potential_clinical_dirs:
        if os.path.exists(potential_dir):
            print(f"✅ Found clinical training directory: {potential_dir}")
            found_clinical_data = True
            
            # Process PDF files, DOCX files, etc. for clinical content
            for root, _, files in os.walk(potential_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    if file.endswith(".pdf"):
                        print(f"📄 Processing clinical PDF: {file} from {os.path.basename(potential_dir)}")
                        raw_text = extract_text_from_pdf(file_path)
                        if raw_text:
                            chunks = chunk_text_by_words(raw_text)
                            valid_chunks = [c for c in chunks if is_valid_chunk(c)]
                            clinical_chunks.extend(valid_chunks)
                            
                    elif file.endswith(".docx"):
                        print(f"📄 Processing clinical DOCX: {file} from {os.path.basename(potential_dir)}")
                        raw_text = extract_text_from_docx(file_path)
                        if raw_text:
                            chunks = chunk_text_by_words(raw_text)
                            valid_chunks = [c for c in chunks if is_valid_chunk(c)]
                            clinical_chunks.extend(valid_chunks)
        else:
            print(f"📁 Directory not found: {potential_dir}")
    
    if not found_clinical_data:
        print(f"⚠️ No clinical training directories found!")
        print("📝 Creating clinical directory structure for future materials...")
        os.makedirs(clinical_training_dir, exist_ok=True)
    
    # ============================
    # 🧭 STEP 2: BUILD NAVIGATION INDEX (ASPS Website Content)
    # ============================
    print("🧭 Processing NAVIGATION content from knowledge bases...")
    
    # Check if we have existing content from JSON files
    scraped_content_dir = os.path.join("org_data", "asps", "extracted_content")
    
    if os.path.exists(scraped_content_dir):
        print(f"✅ Found content directory: {scraped_content_dir}")
        
        # Load navigation-focused knowledge base if available
        nav_kb_file = os.path.join(scraped_content_dir, "navigation_focused_kb.json")
        
        if os.path.exists(nav_kb_file):
            print("📊 Loading navigation-focused knowledge base...")
            with open(nav_kb_file, 'r', encoding='utf-8') as f:
                nav_kb = json.load(f)
            
            for item in nav_kb:
                if 'text' in item and len(item['text'].strip()) > 30:
                    navigation_chunks.append(item['text'])
        else:
            # Fallback: process HTML files for navigation content
            print("📄 Fallback: Processing HTML files for navigation content...")
            html_dir = os.path.join(paths["base"], "html_pages")
            
            if os.path.exists(html_dir):
                from bs4 import BeautifulSoup
                
                for file in os.listdir(html_dir):
                    if not file.endswith(".html"):
                        continue
                    path = os.path.join(html_dir, file)
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            soup = BeautifulSoup(f.read(), "html.parser")
                            
                            # Extract navigation-specific content
                            nav_elements = soup.find_all(['nav', 'header', 'footer'])
                            for nav in nav_elements:
                                text = nav.get_text(separator=" ", strip=True)
                                if len(text.split()) > 10:
                                    navigation_chunks.append(text)
                            
                            # Extract page content for navigation context
                            paragraphs = soup.select("main p, article p")
                            for p in paragraphs:
                                text = p.get_text(separator=" ", strip=True)
                                if len(text.split()) > 20:
                                    navigation_chunks.append(text)
                                    
                    except Exception as e:
                        print(f"⚠️ Failed to parse {file}: {e}")
    
    else:
        print(f"⚠️ Content directory not found: {scraped_content_dir}")
        print("💡 Using basic navigation fallback content...")
        
        # Create some basic navigation content as fallback
        navigation_chunks = [
            "To find information about cosmetic procedures, visit the Cosmetic Procedures section of the ASPS website.",
            "For reconstructive surgery information, check the Reconstructive Procedures section.",
            "To locate a qualified plastic surgeon, use the Find a Surgeon tool on the ASPS website.",
            "Patient safety information is available in the Patient Safety section.",
            "Before and after photos can be found in the Photo Gallery section."
        ]
    
    # ============================
    # 🔢 STEP 3: BUILD FAISS INDEXES
    # ============================
    
    # Build clinical index
    if clinical_chunks:
        print(f"🔢 Building CLINICAL index with {len(clinical_chunks)} chunks...")
        clinical_embeddings = embed_model.encode(clinical_chunks, show_progress_bar=True)
        clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
        clinical_index.add(np.array(clinical_embeddings))
        
        # Save clinical data
        clinical_chunks_path = os.path.join(paths["base"], "clinical_chunks.pkl")
        clinical_embeddings_path = os.path.join(paths["base"], "clinical_embeddings.npy")
        clinical_index_path = os.path.join(paths["base"], "clinical_index.faiss")
        
        with open(clinical_chunks_path, "wb") as f:
            pickle.dump(clinical_chunks, f)
        np.save(clinical_embeddings_path, clinical_embeddings)
        faiss.write_index(clinical_index, clinical_index_path)
        
        print(f"✅ Clinical index saved with {len(clinical_chunks)} chunks")
    else:
        print("⚠️ No clinical chunks found - creating empty clinical index")
        clinical_chunks = ["No clinical training data available"]
        clinical_embeddings = embed_model.encode(clinical_chunks)
        clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
        clinical_index.add(np.array(clinical_embeddings))
    
    # Build navigation index
    if navigation_chunks:
        print(f"🔢 Building NAVIGATION index with {len(navigation_chunks)} chunks...")
        navigation_embeddings = embed_model.encode(navigation_chunks, show_progress_bar=True)
        navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
        navigation_index.add(np.array(navigation_embeddings))
        
        # Save navigation data
        navigation_chunks_path = os.path.join(paths["base"], "navigation_chunks.pkl")
        navigation_embeddings_path = os.path.join(paths["base"], "navigation_embeddings.npy")
        navigation_index_path = os.path.join(paths["base"], "navigation_index.faiss")
        
        with open(navigation_chunks_path, "wb") as f:
            pickle.dump(navigation_chunks, f)
        np.save(navigation_embeddings_path, navigation_embeddings)
        faiss.write_index(navigation_index, navigation_index_path)
        
        print(f"✅ Navigation index saved with {len(navigation_chunks)} chunks")
    else:
        print("⚠️ No navigation chunks found - this shouldn't happen")
        navigation_chunks = ["No navigation data available"]
        navigation_embeddings = embed_model.encode(navigation_chunks)
        navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
        navigation_index.add(np.array(navigation_embeddings))
    
    # ============================
    # 📦 STEP 4: STORE IN GLOBAL MEMORY
    # ============================
    
    # Store in separated format
    ORG_FAISS_INDEXES[org_id] = {
        "clinical": clinical_index,
        "navigation": navigation_index
    }
    ORG_CHUNKS[org_id] = {
        "clinical": clinical_chunks,
        "navigation": navigation_chunks
    }
    ORG_EMBEDDINGS[org_id] = {
        "clinical": clinical_embeddings,
        "navigation": navigation_embeddings
    }
    
    print(f"🎯 Successfully built clinical/navigation indexes for '{org_id}'!")
    print(f"   📚 Clinical chunks: {len(clinical_chunks)}")
    print(f"   🧭 Navigation chunks: {len(navigation_chunks)}")
    
    return {
        "clinical_chunks": len(clinical_chunks),
        "navigation_chunks": len(navigation_chunks)
    }


def load_clinical_navigation_indexes(org_id="asps"):
    """
    Load pre-built clinical and navigation indexes from disk.
    """
    print(f"📥 Loading clinical/navigation indexes for '{org_id}'...")
    paths = get_org_paths(org_id)
    
    try:
        # Load clinical data
        clinical_chunks_path = os.path.join(paths["base"], "clinical_chunks.pkl")
        clinical_embeddings_path = os.path.join(paths["base"], "clinical_embeddings.npy")
        clinical_index_path = os.path.join(paths["base"], "clinical_index.faiss")
        
        with open(clinical_chunks_path, "rb") as f:
            clinical_chunks = pickle.load(f)
        clinical_embeddings = np.load(clinical_embeddings_path)
        clinical_index = faiss.read_index(clinical_index_path)
        
        # Load navigation data
        navigation_chunks_path = os.path.join(paths["base"], "navigation_chunks.pkl")
        navigation_embeddings_path = os.path.join(paths["base"], "navigation_embeddings.npy")
        navigation_index_path = os.path.join(paths["base"], "navigation_index.faiss")
        
        with open(navigation_chunks_path, "rb") as f:
            navigation_chunks = pickle.load(f)
        navigation_embeddings = np.load(navigation_embeddings_path)
        navigation_index = faiss.read_index(navigation_index_path)
        
        # Store in global memory
        ORG_FAISS_INDEXES[org_id] = {
            "clinical": clinical_index,
            "navigation": navigation_index
        }
        ORG_CHUNKS[org_id] = {
            "clinical": clinical_chunks,
            "navigation": navigation_chunks
        }
        ORG_EMBEDDINGS[org_id] = {
            "clinical": clinical_embeddings,
            "navigation": navigation_embeddings
        }
        
        print(f"✅ Loaded clinical/navigation indexes for '{org_id}'")
        print(f"   📚 Clinical chunks: {len(clinical_chunks)}")
        print(f"   🧭 Navigation chunks: {len(navigation_chunks)}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Failed to load existing indexes: {e}")
        return False


# Helper functions for PDF and DOCX extraction
def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files"""
    try:
        import fitz  # PyMuPDF
        with fitz.open(pdf_path) as pdf:
            return "".join([page.get_text() for page in pdf])
    except Exception as e:
        print(f"❌ PDF extraction failed for {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    """Extract text from DOCX files"""
    try:
        from docx import Document
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        print(f"❌ DOCX extraction failed for {docx_path}: {e}")
        return ""

def extract_text_from_image(image_path):
    """
    Extracts text from image files (e.g., PNG, JPG) using Tesseract OCR.
    """
    try:
        from PIL import Image
        import pytesseract
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"❌ Image extraction failed for {image_path}: {e}")
        return ""

def load_training_materials_with_type(training_dir, content_type="clinical", max_words=800):
    """
    Walks through a directory of training material files and returns
    a list of valid training input-output chunks with type tagging.
    
    Parameters:
    - training_dir: Directory containing training materials
    - content_type: Either "clinical" or "navigation" 
    - max_words: Maximum words per chunk
    """
    data = []

    for root, _, files in os.walk(training_dir):
        for file in files:
            file_path = os.path.join(root, file)

            if file.endswith(".pdf"):
                raw_text = extract_text_from_pdf(file_path)
            elif file.endswith(".docx"):
                raw_text = extract_text_from_docx(file_path)
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                raw_text = extract_text_from_image(file_path)
            else:
                continue

            chunks = chunk_text_by_words(raw_text, max_words=max_words)

            for chunk in chunks:
                data.append({
                    "input": f"Summarize: {chunk}",
                    "output": "A summary of the material.",
                    "type": content_type,  # Add type tagging
                    "text": chunk
                })

    logging.info(f"✅ Loaded {len(data)} {content_type} training examples from {training_dir}")
    return pd.DataFrame(data)

def load_training_materials(training_dir, max_words=800):
    """
    Legacy function - now calls the typed version with clinical as default
    """
    return load_training_materials_with_type(training_dir, "clinical", max_words)

# ============================
# 🏗️ ORIGINAL BUILD FUNCTION (LEGACY - NOW REPLACED)
# ============================

def build_faiss_index_from_training_dir(org_id):
    """
    Builds a FAISS index for ASPS organization using locally saved HTML pages.
    """
    paths = get_org_paths(org_id)
    all_chunks = []

    if org_id == "asps":
        logging.info("🌐 Extracting text from ASPS local HTML pages...")
        html_dir = os.path.join(paths["base"], "html_pages")

        if not os.path.exists(html_dir):
            raise ValueError(f"❌ HTML directory not found: {html_dir}")

        from bs4 import BeautifulSoup

        all_text = ""
        for file in os.listdir(html_dir):
            if not file.endswith(".html"):
                continue
            path = os.path.join(html_dir, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    soup = BeautifulSoup(f.read(), "html.parser")
                    paragraphs = soup.select("main p")
                    text_blocks = [
                        p.get_text(separator=" ", strip=True)
                        for p in paragraphs
                        if len(p.get_text(strip=True).split()) > 20
                    ]
                    all_text += "\n\n".join(text_blocks)
            except Exception as e:
                logging.warning(f"⚠️ Failed to parse {file}: {e}")

        if not all_text.strip():
            raise ValueError("❌ No content extracted from ASPS HTML pages.")

        chunks = chunk_text_by_words(all_text)
        valid_chunks = [c for c in chunks if is_valid_chunk(c)]
        all_chunks.extend(valid_chunks)

    else:
        raise ValueError(f"❌ Only 'asps' organization is supported in this demo. Got: {org_id}")

    if not all_chunks:
        raise ValueError(f"❌ No valid text chunks found for org '{org_id}'.")

    # ✅ Encode and index
    logging.info(f"🔢 Encoding {len(all_chunks)} chunks using embedding model...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # 💾 Save everything
    os.makedirs(paths["base"], exist_ok=True)
    with open(paths["chunks_pkl"], "wb") as f:
        pickle.dump(all_chunks, f)
    np.save(paths["embeddings_npy"], embeddings)
    faiss.write_index(index, paths["faiss_index"])

    logging.info(f"✅ FAISS index built and saved for org '{org_id}' with {len(all_chunks)} chunks.")

# ============================
# 📥 15. LOAD FAISS FOR ONE ORG
# ============================

def load_rag_resources(org_id):
    """
    Loads org-specific FAISS index and chunk data.
    Rebuilds if any required files are missing or corrupt.
    """
    global rag_chunks, rag_embeddings, faiss_index
    paths = get_org_paths(org_id)

    try:
        with open(paths["chunks_pkl"], "rb") as f:
            rag_chunks = pickle.load(f)
        rag_embeddings = np.load(paths["embeddings_npy"])
        faiss_index = faiss.read_index(paths["faiss_index"])
        logging.info(f"✅ Loaded FAISS resources for org '{org_id}'")

    except Exception as e:
        logging.warning(f"⚠️ FAISS load failed for org '{org_id}': {e}")
        logging.info("🔧 Rebuilding FAISS index...")
        build_faiss_index_from_training_dir(org_id)

        with open(paths["chunks_pkl"], "rb") as f:
            rag_chunks = pickle.load(f)
        rag_embeddings = np.load(paths["embeddings_npy"])
        faiss_index = faiss.read_index(paths["faiss_index"])
        logging.info(f"✅ Rebuilt FAISS index for org '{org_id}'")

# ============================
# 16. MAIN EXECUTION (CLI MODE)
# ============================

def main():
    global rag_model, faiss_index, rag_chunks, rag_embeddings

    print("🧠 Checking CUDA support:")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))

    print("🚀 Script started.")
    
    # Print deployment checklist for reference
    print_runpod_deployment_checklist()

    # --- Use ASPS org with GitHub knowledge bases ---
    org_id = "asps"

    try:
        print("� STEP 1: Verifying RunPod deployment readiness...")
        deployment_status = verify_clinical_training_setup()
        
        print("\n�📥 STEP 2: GITHUB DEPLOYMENT MODE - Loading pre-scraped knowledge bases...")
        
        # Step 1: Download knowledge bases from GitHub
        github_repo = "swolmer/athena-rag-api"  # ✅ Updated with your actual GitHub repo
        github_branch = "asps_demo"  # ✅ Use the correct branch where files are located
        
        # Try to setup from local git clone files first
        if setup_local_knowledge_bases(org_id):
            print("✅ Local JSON files setup successful!")
            
            # Step 2: Load into memory and build FAISS indexes
            if load_github_knowledge_bases_into_memory(org_id):
                print("✅ Knowledge bases loaded and indexed successfully!")
            else:
                print("❌ Failed to load knowledge bases into memory")
                print("🔄 Falling back to local content and building basic indexes...")
                build_clinical_navigation_indexes(org_id)
                load_clinical_navigation_indexes(org_id)
        else:
            print("❌ GitHub download failed - using local content and basic fallback...")
            print("� Building indexes with available local content...")
            build_clinical_navigation_indexes(org_id)
            if org_id in ORG_FAISS_INDEXES:
                print("✅ Local indexes built successfully!")
            else:
                print("❌ Failed to build local indexes")
                return

    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        traceback.print_exc()
        return

    # --- Optional CLI args for fine-tuning (not used in this ASPS demo) ---
    parser = argparse.ArgumentParser(description="Run ASPS RAG chatbot or evaluation")
    parser.add_argument(
        '--jsonl_path',
        type=str,
        default=os.path.join(BASE_DIR, "step4_structured_instruction_finetune_ready.jsonl"),
        help="Path to JSONL training data (if fine-tuning)"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(BASE_DIR, "Models", "mistral-full-out"),
        help="Directory to save fine-tuned model (if training)"
    )
    args = parser.parse_args()

    model = rag_model  # ✅ Use the loaded model

    # --- 🔍 Sample Evaluation with Questions ---
    sample_questions = [
        # Clinical questions (operative procedures and microsurgical flaps)
        "What are the typical indications for placement of a tissue expander in breast reconstruction surgery?",
        "What is the vascular supply of the radial forearm flap?",
        "Explain the operative steps for a free TRAM flap breast reconstruction.",
        "What artery supplies the vascularized fibula flap?",
        "How is capsulorrhaphy performed during implant exchange in breast reconstruction?",
        "What is the primary blood supply to the TAP flap?",
        "Describe the steps involved in the placement of a tissue expander after mastectomy.",
        "What precautions must be taken to avoid injury to the peroneal nerve during fibula flap harvest?",
        "How is the Allen's test used in the preoperative assessment of the radial forearm flap?",
        "What are the differences between craniofacial and mandibular plates?",
        "What portion of the serratus anterior muscle is typically harvested for the flap?",
        "Why is the distal 6 cm of the fibula preserved during flap harvest?",
        
        # Navigation questions
        "Where can I find information about breast augmentation costs?",
        "How do I locate a plastic surgeon in my area?",
        "Where are the before and after photos on the ASPS website?",
        "How do I navigate to patient safety information?",
        "Where can I read about the risks of cosmetic surgery?",
        "How do I find recovery information for tummy tucks?",
        "Where is the Find a Surgeon tool located?"
    ]

    print("\n📊 Evaluating sample questions using GitHub knowledge bases...")
    evaluate_on_examples(
        model=model,
        tokenizer=tokenizer,
        sample_questions=sample_questions,
        save_path=os.path.join(BASE_DIR, "eval_outputs_github_asps.json"),
        k=3,
        org_id="asps"
    )

    # --- 💬 Chatbot Loop ---
    print("\n🩺 ASPS RAG Chatbot Ready (GitHub Mode). Type a surgical question or 'exit' to quit.")
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("👋 Goodbye!")
                break
            
            if not question:
                print("❓ Please enter a question.")
                continue
            
            # Classify intent and retrieve context
            intent = classify_question_intent(question)
            print(f"🎯 Detected intent: {intent}")
            
            context_chunks = retrieve_context(
                query=question, 
                k=3, 
                org_id=org_id,
                intent=intent
            )
            
            if context_chunks:
                answer = generate_rag_answer_with_context(
                    user_question=question,
                    context_chunks=context_chunks,
                    mistral_tokenizer=tokenizer,
                    mistral_model=model,
                    intent=intent
                )
                print(f"\n🤖 **Answer:**\n{answer}")
            else:
                print("❌ No relevant context found for your question.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error processing question: {e}")
            continue

# ============================
# 17. UPLOAD & INDEX MATERIALS (ORG-AWARE)
# ============================

def handle_uploaded_zip(zip_path: str, org_id: str):
    """
    Accepts a ZIP file path and an org_id.
    Extracts contents to org's training directory and builds FAISS index.
    """
    try:
        # Step 1: Get paths for the org
        paths = get_org_paths(org_id)
        training_dir = paths["training_data_dir"]

        # Step 2: Clear old training data
        if os.path.exists(training_dir):
            shutil.rmtree(training_dir)
        os.makedirs(training_dir, exist_ok=True)

        # Step 3: Extract ZIP contents
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(training_dir)
        logging.info(f"✅ Extracted ZIP for org '{org_id}' to: {training_dir}")

        # Step 4: Build FAISS
        build_faiss_index_from_training_dir(org_id)

        # Step 5: Reload into memory
        load_rag_resources(org_id)

        logging.info(f"🎉 Upload and indexing complete for org '{org_id}'")

    except Exception as e:
       logging.error(f"❌ Failed to load FAISS into memory for org '{org_id}'")


# ============================
# 18. ENTRYPOINT
# ============================

def setup_local_knowledge_bases(org_id="asps"):
    """
    Setup knowledge bases from local git clone files.
    Checks for nav1.json, nav2.json, and navigation_training_data.json
    in the current directory and copies them to the expected location.
    """
    print("📥 GIT CLONE MODE - Using local repository files...")
    
    # Check for local JSON files first (from git clone)
    local_json_files = ["nav1.json", "nav2.json", "navigation_training_data.json"]
    found_files = []
    
    print("🔍 Checking for local JSON files from git clone...")
    for filename in local_json_files:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            found_files.append(filename)
            print(f"   ✅ Found: {filename} ({file_size:.1f} MB)")
        else:
            print(f"   ⚠️ Missing: {filename}")
    
    if len(found_files) >= 2:  # Need at least nav1 and nav2
        print(f"\n✅ Found {len(found_files)} JSON files locally!")
        
        # Copy files to expected location
        paths = get_org_paths(org_id)
        os.makedirs(paths["base"], exist_ok=True)
        
        for filename in found_files:
            source_path = filename
            target_path = os.path.join(paths["base"], filename)
            
            if not os.path.exists(target_path):
                import shutil
                shutil.copy2(source_path, target_path)
                print(f"   📋 Copied {filename} to org_data/asps/")
            else:
                print(f"   ✅ {filename} already in org_data/asps/")
        
        return True
    else:
        print(f"⚠️ Only found {len(found_files)} JSON files locally.")
        return False

# ============================
# 📥 GITHUB KNOWLEDGE BASE LOADER (Legacy - kept for fallback)
# ============================

def download_knowledge_base_from_github(org_id="asps", github_repo="swolmer/athena-rag-api", github_branch="asps_demo"):
    """
    Download pre-extracted knowledge bases from GitHub instead of real-time extraction.
    This is much faster and more reliable than processing content on RunPod.
    
    Downloads knowledge base JSON files for navigation index:
    - navigation_training_data.json: Original navigation training data  
    - nav1.json: Now goes to NAVIGATION index (ALL JSON files → navigation)
    - nav2.json: Now goes to NAVIGATION index (ALL JSON files → navigation)
    
    For private repositories, set GITHUB_TOKEN environment variable.
    For RunPod: This function downloads JSON knowledge bases. Clinical training 
    directories (Training Data Op, Training Data Textbooks, Validate, etc.) 
    should be available in the GitHub repository and will be accessed directly.
    """
    import urllib.request
    import json
    
    print(f"📥 Downloading pre-scraped knowledge bases for '{org_id}' from GitHub...")
    
    # Check if repository is private and needs authentication
    github_token = os.getenv("GITHUB_TOKEN")
    
    # Define GitHub raw URLs for your knowledge base files (directly in repo root)
    github_base_url = f"https://raw.githubusercontent.com/{github_repo}/{github_branch}"
    
    # Files available in your repository (now using simplified nav1/nav2 naming)
    # These will be used for NAVIGATION and CLINICAL index building
    knowledge_files = [
        "navigation_training_data.json",  # ✅ Original navigation data → NAVIGATION INDEX
        "nav1.json",                      # 🆕 ALL JSON files now go to NAVIGATION INDEX
        "nav2.json",                      # 🆕 ALL JSON files now go to NAVIGATION INDEX
        "ultimate_asps_knowledge_base.json"  # 📦 Full file as backup (37.37 MB)
    ]    # Ensure directories exist
    paths = get_org_paths(org_id)
    os.makedirs(paths["base"], exist_ok=True)
    
    downloaded_files = []
    
    try:
        # Download knowledge base files from repo root
        for filename in knowledge_files:
            url = f"{github_base_url}/{filename}"
            local_path = os.path.join(paths["base"], filename)
            
            try:
                print(f"   📡 Downloading {filename}...")
                
                # Create request with authentication if token is available
                req = urllib.request.Request(url)
                if github_token:
                    req.add_header("Authorization", f"token {github_token}")
                    print(f"   🔐 Using GitHub token for private repository access")
                
                # Download the file
                with urllib.request.urlopen(req) as response, open(local_path, 'wb') as f:
                    f.write(response.read())
                
                # Verify file was downloaded and is valid JSON
                if os.path.exists(local_path) and os.path.getsize(local_path) > 100:
                    with open(local_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        downloaded_files.append(filename)
                        print(f"   ✅ {filename}: {len(data)} chunks downloaded")
                else:
                    print(f"   ⚠️ {filename} appears to be empty")
                    
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    print(f"   ❌ {filename} not found (404) - check filename and branch")
                elif e.code == 403:
                    print(f"   ❌ {filename} access denied (403) - repository may be private")
                    if not github_token:
                        print(f"   💡 Set GITHUB_TOKEN environment variable for private repo access")
                else:
                    print(f"   ❌ {filename} HTTP error {e.code}: {e.reason}")
                continue
            except Exception as file_error:
                print(f"   ⚠️ Failed to download {filename}: {file_error}")
                continue
        
        if downloaded_files:
            print(f"🎯 Successfully downloaded {len(downloaded_files)} knowledge base files from GitHub!")
            print(f"   📄 Downloaded: {', '.join(downloaded_files)}")
            return True
        else:
            print("❌ No knowledge base files were successfully downloaded")
            return False
        
    except Exception as e:
        print(f"❌ Failed to download from GitHub: {e}")
        print(f"💡 Make sure your GitHub repo URL is correct and files are uploaded")
        return False

def load_github_knowledge_bases_into_memory(org_id="asps"):
    """
    Load the GitHub-downloaded knowledge bases into memory for FAISS indexing.
    This replaces the content extraction and builds indexes from pre-extracted data.
    """
    print(f"🧠 Loading GitHub knowledge bases into memory for '{org_id}'...")
    
    paths = get_org_paths(org_id)
    
    try:
        clinical_chunks = []
        navigation_chunks = []
        
        # ALL JSON files go to NAVIGATION index (per user specification)
        kb_files = [
            ("navigation_training_data.json", "navigation"),      # Original navigation training data
            ("nav1.json", "navigation"),                         # ALL JSON files → NAVIGATION INDEX
            ("nav2.json", "navigation"),                         # ALL JSON files → NAVIGATION INDEX
            ("ultimate_asps_knowledge_base.json", "navigation"), # ALL JSON files → NAVIGATION INDEX
        ]
        
        # Also load all comprehensive_split_*.json files for navigation
        import glob
        split_files = glob.glob(os.path.join(paths["base"], "comprehensive_split_*.json"))
        for split_file in split_files:
            filename = os.path.basename(split_file)
            kb_files.append((filename, "navigation"))  # ALL comprehensive splits → NAVIGATION INDEX
        
        for filename, content_type in kb_files:
            file_path = os.path.join(paths["base"], filename)
            
            if os.path.exists(file_path):
                print(f"   📄 Loading {filename}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Process chunks based on file type
                    for chunk in data:
                        if isinstance(chunk, str):
                            text = chunk
                        elif isinstance(chunk, dict):
                            text = chunk.get('text', chunk.get('content', ''))
                        else:
                            continue
                        
                        if not text or len(text) < 30:
                            continue
                        
                        # Route to appropriate index
                        if content_type == "navigation":
                            navigation_chunks.append(text)
                        elif content_type == "clinical":
                            clinical_chunks.append(text)
                        elif content_type == "mixed":
                            # For mixed content, try to classify
                            intent = classify_question_intent(text)
                            if intent == "clinical":
                                clinical_chunks.append(text)
                            else:
                                navigation_chunks.append(text)
                    
                    print(f"   ✅ Processed {len(data)} chunks from {filename}")
                    
                except Exception as file_error:
                    print(f"   ⚠️ Error processing {filename}: {file_error}")
            else:
                print(f"   ⚠️ {filename} not found, skipping...")
        
        # ============================
        # 📚 STEP 2: LOAD CLINICAL TRAINING DIRECTORIES (ONLY 3 BLUE FOLDERS!)
        # ============================
        print("📚 Loading clinical training directories for CLINICAL FAISS...")
        
        # ONLY the 3 blue folders specified by user for CLINICAL content
        clinical_training_dirs = [
            "clinical",         # Blue folder 1
            "op notes",         # Blue folder 2  
            "textbook notes"    # Blue folder 3
        ]
        
        # Look for clinical directories in base path (after GitHub clone)
        for dir_name in clinical_training_dirs:
            # First try in org_data/asps/ (copied location)
            potential_dir = os.path.join(paths["base"], dir_name)
            
            # If not found there, try in the repository root (current working directory)
            if not os.path.exists(potential_dir):
                potential_dir = os.path.join(os.getcwd(), dir_name)
            
            if os.path.exists(potential_dir):
                print(f"✅ Found clinical directory: {dir_name}")
                
                # Process all PDF and DOCX files in this directory
                for root, _, files in os.walk(potential_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        
                        if file.endswith(".pdf"):
                            print(f"   📄 Processing clinical PDF: {file}")
                            raw_text = extract_text_from_pdf(file_path)
                            if raw_text:
                                chunks = chunk_text_by_words(raw_text)
                                valid_chunks = [c for c in chunks if is_valid_chunk(c)]
                                clinical_chunks.extend(valid_chunks)
                                print(f"      Added {len(valid_chunks)} chunks from {file}")
                        
                        elif file.endswith(".docx"):
                            print(f"   📝 Processing clinical DOCX: {file}")
                            raw_text = extract_text_from_docx(file_path)
                            if raw_text:
                                chunks = chunk_text_by_words(raw_text)
                                valid_chunks = [c for c in chunks if is_valid_chunk(c)]
                                clinical_chunks.extend(valid_chunks)
                                print(f"      Added {len(valid_chunks)} chunks from {file}")
            else:
                print(f"⚠️ Clinical directory not found: {dir_name} (checked both org_data/asps/ and repository root)")
        
        # Remove duplicates while preserving order
        clinical_chunks = list(dict.fromkeys(clinical_chunks))
        navigation_chunks = list(dict.fromkeys(navigation_chunks))
        
        print(f"📊 Final processed knowledge base:")
        print(f"   🧭 Navigation chunks: {len(navigation_chunks)} (ALL JSON files)")
        print(f"   📚 Clinical chunks: {len(clinical_chunks)} (3 blue folders only)")
        print(f"   📋 Data distribution:")
        print(f"      Navigation: nav1.json + nav2.json + navigation_training_data.json + comprehensive_split_*.json")
        print(f"      Clinical: clinical/ + op notes/ + textbook notes/ folders")
        
        # Build FAISS indexes
        print(f"🔢 Building FAISS indexes...")
        
        # Clinical index
        if clinical_chunks:
            print("   🧠 Building clinical index...")
            clinical_embeddings = embed_model.encode(clinical_chunks, show_progress_bar=True)
            clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
            clinical_index.add(np.array(clinical_embeddings))
        else:
            print("⚠️ No clinical chunks found - creating minimal index")
            clinical_chunks = ["Clinical information will be available soon."]
            clinical_embeddings = embed_model.encode(clinical_chunks)
            clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
            clinical_index.add(np.array(clinical_embeddings))
        
        # Navigation index
        if navigation_chunks:
            print("   🧭 Building navigation index...")
            navigation_embeddings = embed_model.encode(navigation_chunks, show_progress_bar=True)
            navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
            navigation_index.add(np.array(navigation_embeddings))
        else:
            print("⚠️ No navigation chunks found - creating minimal index")
            navigation_chunks = ["Navigation information will be available soon."]
            navigation_embeddings = embed_model.encode(navigation_chunks)
            navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
            navigation_index.add(np.array(navigation_embeddings))
        
        # Store in global memory (dual index format)
        ORG_FAISS_INDEXES[org_id] = {
            "clinical": clinical_index,
            "navigation": navigation_index
        }
        ORG_CHUNKS[org_id] = {
            "clinical": clinical_chunks,
            "navigation": navigation_chunks
        }
        ORG_EMBEDDINGS[org_id] = {
            "clinical": clinical_embeddings,
            "navigation": navigation_embeddings
        }
        
        print(f"✅ Successfully loaded GitHub knowledge bases into memory!")
        print(f"🎯 Ready for clinical and navigation queries!")
        
        return True
        
        print(f"📊 Processed knowledge base:")
        print(f"   📚 Clinical chunks: {len(clinical_chunks)}")
        print(f"   🧭 Navigation chunks: {len(navigation_chunks)}")
        
        # Build FAISS indexes
        print(f"🔢 Building FAISS indexes...")
        
        # Clinical index
        if clinical_chunks:
            clinical_embeddings = embed_model.encode(clinical_chunks, show_progress_bar=True)
            clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
            clinical_index.add(np.array(clinical_embeddings))
        else:
            print("⚠️ No clinical chunks found")
            clinical_chunks = ["No clinical data available"]
            clinical_embeddings = embed_model.encode(clinical_chunks)
            clinical_index = faiss.IndexFlatL2(clinical_embeddings.shape[1])
            clinical_index.add(np.array(clinical_embeddings))
        
        # Navigation index
        if navigation_chunks:
            navigation_embeddings = embed_model.encode(navigation_chunks, show_progress_bar=True)
            navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
            navigation_index.add(np.array(navigation_embeddings))
        else:
            print("⚠️ No navigation chunks found")
            navigation_chunks = ["No navigation data available"]
            navigation_embeddings = embed_model.encode(navigation_chunks)
            navigation_index = faiss.IndexFlatL2(navigation_embeddings.shape[1])
            navigation_index.add(np.array(navigation_embeddings))
        
        # Store in global memory (dual index format)
        ORG_FAISS_INDEXES[org_id] = {
            "clinical": clinical_index,
            "navigation": navigation_index
        }
        ORG_CHUNKS[org_id] = {
            "clinical": clinical_chunks,
            "navigation": navigation_chunks
        }
        ORG_EMBEDDINGS[org_id] = {
            "clinical": clinical_embeddings,
            "navigation": navigation_embeddings
        }
        
        print(f"✅ Successfully loaded GitHub knowledge bases into memory!")
        print(f"🎯 Ready for clinical and navigation queries!")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load GitHub knowledge bases: {e}")
        traceback.print_exc()
        return False

# ============================
# 🌐 FASTAPI DEMO ENDPOINTS
# ============================

# FastAPI imports already handled at the top of the file

app = FastAPI(title="ASPS RAG Demo API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system on startup
@app.on_event("startup")
async def startup_event():
    """Initialize the ASPS RAG system when the API starts"""
    print("🚀 FastAPI startup - initializing ASPS system...")
    initialize_asps_system()

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    k: int = 3

class QueryResponse(BaseModel):
    answer: str
    context_chunks: list

@app.get("/", response_class=HTMLResponse)
async def chatbot_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🩺 ASPS Medical AI Chatbot</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh; display: flex; align-items: center; justify-content: center;
            }
            .chat-container {
                width: 900px; height: 700px; background: white; border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3); display: flex; flex-direction: column;
                overflow: hidden;
            }
            .chat-header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                padding: 20px; color: white; text-align: center;
            }
            .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
            .chat-header p { opacity: 0.9; font-size: 14px; }
            .chat-messages {
                flex: 1; padding: 20px; overflow-y: auto; background: #f8f9fa;
            }
            .message {
                margin-bottom: 15px; display: flex; align-items: flex-start;
            }
            .message.user { justify-content: flex-end; }
            .message-content {
                max-width: 70%; padding: 12px 16px; border-radius: 18px;
                word-wrap: break-word;
            }
            .message.user .message-content {
                background: #007bff; color: white; border-bottom-right-radius: 4px;
            }
            .message.bot .message-content {
                background: white; color: #333; border: 1px solid #e0e0e0;
                border-bottom-left-radius: 4px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .chat-input-container {
                padding: 20px; background: white; border-top: 1px solid #e0e0e0;
                display: flex; gap: 10px;
            }
            .chat-input {
                flex: 1; padding: 12px 16px; border: 2px solid #e0e0e0;
                border-radius: 25px; font-size: 14px; outline: none;
                transition: border-color 0.3s;
            }
            .chat-input:focus { border-color: #007bff; }
            .send-button {
                padding: 12px 24px; background: #007bff; color: white;
                border: none; border-radius: 25px; cursor: pointer;
                font-weight: 600; transition: background 0.3s;
            }
            .send-button:hover { background: #0056b3; }
            .send-button:disabled { background: #ccc; cursor: not-allowed; }
            .typing-indicator {
                display: none; padding: 12px 16px; background: white;
                border: 1px solid #e0e0e0; border-radius: 18px; border-bottom-left-radius: 4px;
                max-width: 70%;
            }
            .typing-dots { display: flex; gap: 4px; }
            .typing-dots span {
                width: 8px; height: 8px; background: #999; border-radius: 50%;
                animation: typing 1.4s infinite ease-in-out;
            }
            .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
            .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
            @keyframes typing {
                0%, 80%, 100% { transform: scale(0); }
                40% { transform: scale(1); }
            }
            .sample-questions {
                padding: 15px; background: #f0f8ff; border-radius: 10px;
                margin-bottom: 20px;
            }
            .sample-questions h3 { margin-bottom: 10px; color: #333; font-size: 16px; }
            .question-button {
                display: inline-block; margin: 5px; padding: 8px 12px;
                background: #e3f2fd; color: #1976d2; border: 1px solid #1976d2;
                border-radius: 15px; cursor: pointer; font-size: 12px;
                transition: all 0.3s;
            }
            .question-button:hover {
                background: #1976d2; color: white;
            }
            .status-indicator {
                position: absolute; top: 10px; right: 10px; padding: 5px 10px;
                border-radius: 10px; font-size: 12px; font-weight: bold;
            }
            .status-online { background: #d4edda; color: #155724; }
            .status-offline { background: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="chat-container">
            <div class="chat-header">
                <div class="status-indicator" id="statusIndicator">🔄 Connecting...</div>
                <h1>🩺 ASPS Medical AI Assistant</h1>
                <p>Clinical knowledge & website navigation powered by Mistral-7B</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="sample-questions">
                    <h3>💡 Try asking about:</h3>
                    <strong>Clinical Questions (from your operative training materials):</strong><br>
                    <div class="question-button" onclick="askQuestion('What are the typical indications for placement of a tissue expander in breast reconstruction surgery?')">Tissue expander indications</div>
                    <div class="question-button" onclick="askQuestion('What is the vascular supply of the radial forearm flap?')">Radial forearm flap anatomy</div>
                    <div class="question-button" onclick="askQuestion('Explain the operative steps for a free TRAM flap breast reconstruction.')">Free TRAM flap procedure</div>
                    <div class="question-button" onclick="askQuestion('What artery supplies the vascularized fibula flap?')">Fibula flap vascular supply</div>
                    <div class="question-button" onclick="askQuestion('How is capsulorrhaphy performed during implant exchange in breast reconstruction?')">Capsulorrhaphy technique</div>
                    <div class="question-button" onclick="askQuestion('What is the primary blood supply to the TAP flap?')">TAP flap blood supply</div>
                    <br><strong>Navigation Questions:</strong><br>
                    <div class="question-button" onclick="askQuestion('How do I use the Find a Surgeon tool on plasticsurgery.org?')">How do I use Find a Surgeon tool?</div>
                    <div class="question-button" onclick="askQuestion('What specific steps do I take to verify a plastic surgeon is board certified?')">How to verify board certification?</div>
                    <div class="question-button" onclick="askQuestion('Where exactly on plasticsurgery.org can I see breast augmentation before and after photos?')">Where to see before/after photos?</div>
                </div>
                
                <div class="message bot">
                    <div class="message-content">
                        👋 Welcome! I'm your ASPS medical assistant. I can help with:<br><br>
                        🩺 <strong>Medical Questions:</strong> Plastic surgery procedures, techniques, and clinical knowledge from your training materials<br>
                        🧭 <strong>Website Navigation:</strong> Finding information on plasticsurgery.org, locating tools, photos, surgeon directories, and resources<br><br>
                        What would you like to know?
                    </div>
                </div>
                
                <div class="typing-indicator" id="typingIndicator">
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            </div>
            
            <div class="chat-input-container">
                <input type="text" id="chatInput" class="chat-input" 
                       placeholder="Ask about plastic surgery procedures..." 
                       onkeypress="handleKeyPress(event)">
                <button id="sendButton" class="send-button" onclick="sendMessage()">Send</button>
            </div>
        </div>

        <script>
            let isWaitingForResponse = false;
            const apiUrl = window.location.origin;

            // Check server status on load
            checkServerStatus();

            function checkServerStatus() {
                fetch(apiUrl + '/health')
                    .then(response => response.json())
                    .then(data => {
                        const indicator = document.getElementById('statusIndicator');
                        if (data.org_loaded) {
                            indicator.textContent = '✅ Online';
                            indicator.className = 'status-indicator status-online';
                        } else {
                            indicator.textContent = '⚠️ Loading';
                            indicator.className = 'status-indicator status-offline';
                        }
                    })
                    .catch(error => {
                        const indicator = document.getElementById('statusIndicator');
                        indicator.textContent = '❌ Offline';
                        indicator.className = 'status-indicator status-offline';
                    });
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    sendMessage();
                }
            }

            function askQuestion(question) {
                document.getElementById('chatInput').value = question;
                sendMessage();
            }

            function addMessage(content, isUser = false) {
                const messagesContainer = document.getElementById('chatMessages');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user' : 'bot'}`;
                
                const contentDiv = document.createElement('div');
                contentDiv.className = 'message-content';
                contentDiv.innerHTML = content;
                
                messageDiv.appendChild(contentDiv);
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }

            function showTyping() {
                document.getElementById('typingIndicator').style.display = 'block';
                document.getElementById('chatMessages').scrollTop = document.getElementById('chatMessages').scrollHeight;
            }

            function hideTyping() {
                document.getElementById('typingIndicator').style.display = 'none';
            }

            async function sendMessage() {
                if (isWaitingForResponse) return;

                const input = document.getElementById('chatInput');
                const message = input.value.trim();
                
                if (!message) return;

                // Add user message
                addMessage(message, true);
                input.value = '';
                
                // Show typing indicator
                showTyping();
                isWaitingForResponse = true;
                document.getElementById('sendButton').disabled = true;

                try {
                    const response = await fetch(apiUrl + '/query', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: message,
                            k: 3
                        })
                    });

                    hideTyping();

                    if (response.ok) {
                        const data = await response.json();
                        addMessage(formatAnswer(data.answer));
                    } else {
                        const errorData = await response.json();
                        addMessage(`❌ Error: ${errorData.detail || 'Unknown error occurred'}`);
                    }
                } catch (error) {
                    hideTyping();
                    addMessage(`❌ Connection error: ${error.message}. Please check if the server is running.`);
                } finally {
                    isWaitingForResponse = false;
                    document.getElementById('sendButton').disabled = false;
                    input.focus();
                }
            }

            function formatAnswer(answer) {
                // Format both clinical and navigation answers with better styling
                return answer
                    // Clinical format
                    .replace(/✅ Summary:/g, '<strong>✅ Summary:</strong>')
                    .replace(/🧠 Anatomy & Physiology:/g, '<br><strong>🧠 Anatomy & Physiology:</strong>')
                    .replace(/🔧 Procedure or Technique:/g, '<br><strong>🔧 Procedure or Technique:</strong>')
                    .replace(/⚠️ Pitfalls & Pearls:/g, '<br><strong>⚠️ Pitfalls & Pearls:</strong>')
                    // Navigation format
                    .replace(/📍 Direct Answer:/g, '<strong>📍 Direct Answer:</strong>')
                    .replace(/🔗 Where to Find:/g, '<br><strong>🔗 Where to Find:</strong>')
                    .replace(/💡 Additional Help:/g, '<br><strong>💡 Additional Help:</strong>')
                    .replace(/\\n/g, '<br>');
            }

            // Check status every 30 seconds
            setInterval(checkServerStatus, 30000);
        </script>
    </body>
    </html>
    """

@app.get("/api")
async def api_info():
    return {"message": "🩺 ASPS RAG Demo API is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
        "org_loaded": "asps" in ORG_FAISS_INDEXES
    }

@app.post("/query", response_model=QueryResponse)
async def query_asps_rag(request: QueryRequest):
    """Query the ASPS RAG system with clinical/navigation intent detection"""
    try:
        # Ensure ASPS data is loaded
        if "asps" not in ORG_FAISS_INDEXES:
            raise HTTPException(
                status_code=503, 
                detail="ASPS data not loaded. Please wait for system initialization."
            )
        
        # Classify the question intent
        intent = classify_question_intent(request.question)
        print(f"🎯 Question: '{request.question[:50]}...' -> Intent: {intent}")
        
        # Retrieve context based on intent
        context_chunks = retrieve_context(
            query=request.question,
            k=request.k,
            org_id="asps",
            intent=intent
        )
        
        # Generate answer
        answer = generate_rag_answer_with_context(
            user_question=request.question,
            context_chunks=context_chunks,
            mistral_tokenizer=tokenizer,
            mistral_model=rag_model,
            intent=intent
        )
        
        # Add knowledge type indicator for user transparency
        enhanced_answer = f"[Using {intent.upper()} knowledge] {answer}"
        
        return QueryResponse(
            answer=enhanced_answer,  # Include knowledge type indicator
            context_chunks=context_chunks
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.post("/api/chat", response_model=QueryResponse)
async def query_chat_alias(request: QueryRequest):
    """Alias endpoint to support frontend expecting /api/chat instead of /query"""
    return await query_asps_rag(request)

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions to demo the system"""
    return {
        "questions": [
            "What are the typical indications for placement of a tissue expander in breast reconstruction surgery?",
            "What is the vascular supply of the radial forearm flap?",
            "Explain the operative steps for a free TRAM flap breast reconstruction.",
            "What artery supplies the vascularized fibula flap?",
            "How is capsulorrhaphy performed during implant exchange in breast reconstruction?",
            "What is the primary blood supply to the TAP flap?",
            "Describe the steps involved in the placement of a tissue expander after mastectomy.",
            "What precautions must be taken to avoid injury to the peroneal nerve during fibula flap harvest?",
            "How is the Allen's test used in the preoperative assessment of the radial forearm flap?",
            "What are the differences between craniofacial and mandibular plates?"
        ]
    }

def initialize_asps_system():
    """Initialize the ASPS RAG system with clinical/navigation separation"""
    print("🚀 Initializing ASPS RAG system with clinical/navigation separation...")
    
    # Step 0: Verify clinical training materials are accessible
    verify_clinical_training_setup()
    
    org_id = "asps"
    
    try:
        print("📥 STEP 1: Checking for local JSON files (git clone approach)...")
        
        # Try local git clone files first (preferred method)
        if setup_local_knowledge_bases(org_id):
            print("✅ Local JSON files found and copied!")
            
            # Load the JSON knowledge bases into memory and build indexes
            if load_github_knowledge_bases_into_memory(org_id):
                print("✅ Knowledge bases loaded and indexed successfully!")
            else:
                print("⚠️ Failed to load JSON knowledge bases - building fallback indexes...")
                build_clinical_navigation_indexes(org_id)
                load_clinical_navigation_indexes(org_id)
        else:
            print("⚠️ No local JSON files found - trying GitHub download...")
            
            # Fallback to GitHub download
            if download_knowledge_base_from_github(org_id):
                print("✅ GitHub download successful!")
                if load_github_knowledge_bases_into_memory(org_id):
                    print("✅ Knowledge bases loaded and indexed successfully!")
                else:
                    print("⚠️ Failed to load downloaded files - building fallback indexes...")
                    build_clinical_navigation_indexes(org_id)
                    load_clinical_navigation_indexes(org_id)
            else:
                print("⚠️ GitHub download failed - building basic fallback indexes...")
                
                # Try to load existing indexes first
                if load_clinical_navigation_indexes(org_id):
                    print("✅ Loaded existing clinical/navigation indexes!")
                else:
                    print("🏗️ Building new clinical/navigation indexes from local content...")
                    result = build_clinical_navigation_indexes(org_id)
                    print(f"✅ Built new indexes:")
                    print(f"   📚 Clinical chunks: {result['clinical_chunks']}")
                    print(f"   🧭 Navigation chunks: {result['navigation_chunks']}")
        
        # Verify the final system state
        if org_id in ORG_FAISS_INDEXES:
            clinical_count = len(ORG_CHUNKS[org_id].get("clinical", []))
            navigation_count = len(ORG_CHUNKS[org_id].get("navigation", []))
            total_count = clinical_count + navigation_count
            
            print(f"🎯 ASPS RAG System Status:")
            print(f"   📚 Clinical knowledge: {clinical_count} chunks")
            print(f"   🧭 Navigation knowledge: {navigation_count} chunks") 
            print(f"   🔥 Total knowledge base: {total_count} chunks")
            print(f"   ✅ System ready for clinical/navigation queries!")
            
            # Log the approach that worked
            if clinical_count > 1000:  # Indicates successful JSON loading
                print("   🎖️ Using comprehensive JSON knowledge bases")
            else:
                print("   🔄 Using local content fallback")
        else:
            raise RuntimeError("Failed to initialize indexes")
        
    except Exception as e:
        print(f"❌ Failed to initialize ASPS RAG system: {e}")
        import traceback
        traceback.print_exc()
        # Continue without crashing - the health endpoint will show the error

if __name__ == "__main__":
    print("🏥 ASPS MEDICAL AI CHATBOT - CLINICAL/NAVIGATION DUAL SYSTEM")
    print("=" * 70)
    print("🎯 System Overview:")
    print("   • CLINICAL questions → Training materials FAISS index")
    print("   • NAVIGATION questions → Website knowledge FAISS index")  
    print("   • Automatic intent detection and routing")
    print("   • Mistral-7B powered medical responses")
    print("")
    print("📊 Data Sources:")
    print("   📚 Clinical: PDFs/DOCX in clinical_training/ directory")
    print("   🧭 Navigation: ASPS website content via knowledge bases")
    print("")
    print("🔧 Key Components:")
    print("   🎯 classify_question_intent() - Routes questions")
    print("   📚 build_clinical_navigation_indexes() - Dual FAISS setup")
    print("   🔍 retrieve_context() - Intent-based retrieval")
    print("   🤖 generate_rag_answer_with_context() - Response generation")
    print("")
    print("💡 Example Questions:")
    print("   Clinical: 'What are the key operative techniques for breast reconstruction?'")
    print("   Navigation: 'How do I use the Find a Surgeon tool on plasticsurgery.org?'")
    print("")
    
    # Initialize ASPS system
    initialize_asps_system()
    
    # Start FastAPI server
    print("\n🌐 Starting ASPS Demo API server...")
    print("📍 Server will be available at: http://213.173.110.81:19524")
    print("📖 API docs at: http://213.173.110.81:19524/docs")
    print("🏥 Health check at: http://213.173.110.81:19524/health")
    print("🔗 RunPod External Access: Connect via TCP port 213.173.110.81:19524")
    print("\n✅ SYSTEM READY FOR CLINICAL/NAVIGATION DIFFERENTIATION!")
    uvicorn.run(app, host="0.0.0.0", port=19524)

# ============================
# 📋 SYSTEM DOCUMENTATION
# ============================
"""
🏥 ASPS MEDICAL AI CHATBOT - DUAL KNOWLEDGE SYSTEM

PURPOSE:
This system creates a medical AI chatbot that intelligently differentiates between:
- CLINICAL questions (medical procedures, risks, techniques) 
- NAVIGATION questions (finding surgeons, costs, appointments)

ARCHITECTURE:
1. Intent Classification:
   - classify_question_intent() analyzes user questions
   - Returns "clinical" or "navigation" based on keywords/context
   
2. Dual FAISS Indexes:
   - Clinical Index: Built from training materials (PDFs, DOCX)
   - Navigation Index: Built from ASPS website knowledge data
   
3. Context Retrieval:
   - retrieve_context() searches appropriate index based on intent
   - Returns relevant chunks for answer generation
   
4. Response Generation:
   - generate_rag_answer_with_context() uses Mistral-7B
   - Provides structured medical answers with proper context

DATA FLOW:
User Question → Intent Classification → Index Selection → Context Retrieval → Answer Generation

SETUP REQUIREMENTS:
1. Add clinical training materials to: org_data/asps/clinical_training/
2. Run with JSON knowledge bases for navigation content
3. Execute: python demo_asps.py
4. Access web interface at: http://localhost:19524

FILES CREATED:
- Clinical FAISS index from training materials  
- Navigation FAISS index from website knowledge
- Dual chunk storage with intent separation
- Web API with automatic routing

TESTING:
- Clinical: "What is rhinoplasty recovery like?"
- Navigation: "How much does a nose job cost?"
- System automatically routes to correct knowledge base

DEPLOYMENT:
Ready for RunPod deployment with automatic initialization.
All dependencies in requirements.txt.
"""