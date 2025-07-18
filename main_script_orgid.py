# ============================
# 🚀 1. IMPORTS & GLOBAL STORAGE
# ============================

# --- Standard library ---
import os
import json
import logging
import pickle

# --- Numerical / Data ---
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# --- Environment Variables ---
from dotenv import load_dotenv
load_dotenv()  # ✅ Load environment variables from .env file

HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token")
RAG_API_KEY = os.getenv("RAG_API_KEY", "kilment1234")

if not RAG_API_KEY:
    print("⚠️ RAG_API_KEY is not set. Check your .env file.")

# --- Natural Language Processing ---
import nltk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer

nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path, quiet=True)
nltk.data.path.append(nltk_data_path)

def safe_sent_tokenize(text, lang='english'):
    try:
        punkt_path = nltk.data.find(f'tokenizers/punkt/{lang}.pickle')
        with open(punkt_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.tokenize(text)
    except Exception as e:
        print(f"❌ NLTK sent_tokenize fallback used due to: {e}")
        return text.split('.')  # Fallback method

# --- Transformers and Trainer ---
from transformers import (
    AutoConfig,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments,
    AutoTokenizer,
    AutoModelForCausalLM
)

# --- File Extraction Utilities ---
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document

# --- Embedding & FAISS ---
from sentence_transformers import SentenceTransformer
import faiss

# --- Evaluation ---
from evaluate import load as load_metric

# ✅ CUDA Setup
print("🧠 Checking CUDA support:")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No CUDA-compatible GPU detected. Training will run on CPU.")

# ============================
# 🧠 2. GLOBAL ORG-SPECIFIC STORAGE
# ============================

# These store model data per organization
ORG_FAISS_INDEXES = {}     # org_id → FAISS index
ORG_CHUNKS = {}            # org_id → list of document chunks
ORG_EMBEDDINGS = {}        # org_id → np.ndarray of embeddings

# ✅ Tokenizer & Model will be shared
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
    """
    base = os.path.join(ORG_DATA_ROOT, org_id)
    return {
        "base": base,
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
# 7. UTILITIES
# ============================

def is_valid_chunk(text):
    """
    Determines whether a chunk should be included in the index.
    Filters out headers, copyright pages, watermarks, etc.
    """
    text_lower = text.lower()
    skip_phrases = [
        "table of contents",
        "for additional online content",
        "copyright",
        "chang gung",
        "this page intentionally left blank",
        "http://", "https://"
    ]
    skip_starts = [
        "figure", "edition", "flaps and", "samir mardini"
    ]

    if len(text.split()) <= 20:
        return False
    if any(phrase in text_lower for phrase in skip_phrases):
        return False
    if any(text_lower.strip().startswith(start) for start in skip_starts):
        return False

    return True


def chunk_text_by_words(text, max_words=200, overlap=50, min_words=30):
    """
    Splits raw text into overlapping, word-based chunks suitable for retrieval.
    Filters out low-quality chunks.
    """
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

    # Final chunk
    if len(current_chunk) >= min_words:
        final_chunk = " ".join(current_chunk).strip()
        if is_valid_chunk(final_chunk):
            chunks.append(final_chunk)

    return chunks


def extract_text_from_pdf(pdf_path):
    """
    Extracts text from all pages of a PDF file.
    """
    try:
        with fitz.open(pdf_path) as pdf:
            return "".join([page.get_text() for page in pdf])
    except Exception as e:
        logging.error(f"❌ PDF extraction failed for {pdf_path}: {e}")
        return ""


def extract_text_from_image(image_path):
    """
    Extracts text from image files (e.g., PNG, JPG) using Tesseract OCR.
    """
    try:
        image = Image.open(image_path)
        return pytesseract.image_to_string(image)
    except Exception as e:
        logging.error(f"❌ Image extraction failed for {image_path}: {e}")
        return ""


def extract_text_from_docx(docx_path):
    """
    Extracts text from Microsoft Word (.docx) files.
    """
    try:
        doc = Document(docx_path)
        return "\n".join([p.text for p in doc.paragraphs])
    except Exception as e:
        logging.error(f"❌ DOCX extraction failed for {docx_path}: {e}")
        return ""


def load_training_materials(training_dir, max_words=800):
    """
    Walks through a directory of training material files and returns
    a list of valid training input-output chunks.
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
                    "output": "A summary of the material."
                })

    logging.info(f"✅ Loaded {len(data)} training examples from {training_dir}")
    return pd.DataFrame(data)

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

def retrieve_context(query, k=3, initial_k=10, org_id=None, collab=False):
    """
    Retrieves k most relevant chunks.
    If collab=True, you could merge chunks from multiple orgs (not implemented here).
    If collab=False, retrieves from specified org_id.
    """
    if not query or not hasattr(embed_model, 'encode'):
        raise ValueError("❌ Query is empty or embed_model not initialized.")

    if collab:
        raise NotImplementedError("Collaborative retrieval is not implemented yet.")
    if not org_id:
        raise ValueError("❌ 'org_id' must be provided if collab=False.")

    # Load per-org data
    faiss_index = ORG_FAISS_INDEXES.get(org_id)
    rag_chunks = ORG_CHUNKS.get(org_id)
    rag_embeddings = ORG_EMBEDDINGS.get(org_id)

    if (
        faiss_index is None or
        rag_chunks is None or len(rag_chunks) == 0 or
        rag_embeddings is None or len(rag_embeddings) == 0
    ):
        raise ValueError(f"❌ FAISS data for org_id '{org_id}' not loaded.")

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
        from sklearn.metrics.pairwise import cosine_similarity
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
        logging.error(f"❌ Failed to retrieve context: {e}")
        return []

# ============================
# 11. RAG GENERATION — FIXED
# ============================

def generate_rag_answer_with_context(user_question, context_chunks, mistral_tokenizer, mistral_model):
    import re
    from collections import Counter
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    if not context_chunks:
        return "⚠️ No relevant context found to answer this question."

    context = "\n\n".join(f"- {chunk.strip()}" for chunk in context_chunks)
    
    prompt = (
        "You are a surgical expert writing answers for a clinical reference guide.\n"
        "Use only the CONTEXT below to answer the QUESTION in a structured format:\n\n"
        "✅ Summary: (1 sentence)\n"
        "🧠 Anatomy & Physiology:\n"
        "🔧 Procedure or Technique:\n"
        "⚠️ Pitfalls & Pearls:\n\n"
        f"### CONTEXT:\n{context}\n\n"
        f"### QUESTION:\n{user_question}\n\n"
        f"### ANSWER:\n"
    )

    inputs = mistral_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(mistral_model.device)

    with torch.no_grad():
        outputs = mistral_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=128,     # much faster
            do_sample=False,        # deterministic, faster
            eos_token_id=mistral_tokenizer.eos_token_id,
            pad_token_id=mistral_tokenizer.pad_token_id
)

    decoded = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract portion after '### ANSWER:'
    if "### ANSWER:" in decoded:
        answer = decoded.split("### ANSWER:")[-1].strip()
    else:
        answer = decoded.strip()

    # ✂️ Truncate after 6 sentences (to avoid repetition loops)
    answer = re.split(r'\.\s+', answer, maxsplit=6)
    answer = '. '.join(answer).strip()
    if not answer.endswith("."):
        answer += "."

    # 🚨 Add hallucination filter: check token overlap with context
    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    overlap = answer_tokens & context_tokens
    overlap_score = len(overlap) / max(1, len(answer_tokens))

    if overlap_score < 0.35:
        logging.warning("⚠️ Low token overlap — likely hallucination.")
        return "⚠️ Unable to generate a confident answer from the provided surgical materials."

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
            # Step 1: Retrieve top-k chunks
            context_chunks = retrieve_context(query=question, k=k, org_id=org_id)
            context_combined = " ".join(context_chunks)

            # Step 2: Generate answer from model
            answer = generate_rag_answer_with_context(
                user_question=question,
                context_chunks=context_chunks,
                mistral_tokenizer=tokenizer,
                mistral_model=model
            )

            # Step 3: Compute token overlap
            overlap_score = token_overlap_score(answer, context_combined)
            hallucinated = overlap_score < 0.35

            if hallucinated:
                logging.warning(f"⚠️ Token Overlap = {overlap_score:.2f} — potential hallucination.")
                answer = "⚠️ Unable to generate a confident answer from the available context."

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
# 📦 13. ZIP FILE HANDLER (UPLOAD + INDEX)
# ============================

import zipfile
import shutil

def handle_uploaded_zip(zip_path, org_id):
    """
    Handles a ZIP upload:
    - Extracts to org's training folder
    - Builds FAISS index from the contents
    - Loads index into memory for RAG retrieval
    """
    paths = get_org_paths(org_id)
    training_dir = paths["training_data_dir"]

    # 🧹 Clean old training data (optional safety)
    if os.path.exists(training_dir):
        shutil.rmtree(training_dir)
    os.makedirs(training_dir, exist_ok=True)

    # 📂 Unzip to training directory
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(training_dir)
        logging.info(f"✅ Extracted ZIP to: {training_dir}")
    except Exception as e:
        logging.error(f"❌ Failed to extract ZIP: {e}")
        return

    # 🔨 Build and load FAISS index
    try:
        build_faiss_index_from_training_dir(org_id)
        load_faiss_into_memory(org_id)
        logging.info(f"✅ ZIP upload + indexing complete for org '{org_id}'")
    except Exception as e:
        logging.error(f"❌ Failed to build/load FAISS after upload: {e}")


def load_faiss_into_memory(org_id):
    """
    Loads the org-specific index/chunks/embeddings into the global lookup dicts.
    This makes RAG work immediately after ZIP upload.
    """
    paths = get_org_paths(org_id)
    try:
        with open(paths["chunks_pkl"], "rb") as f:
            ORG_CHUNKS[org_id] = pickle.load(f)
        ORG_EMBEDDINGS[org_id] = np.load(paths["embeddings_npy"])
        ORG_FAISS_INDEXES[org_id] = faiss.read_index(paths["faiss_index"])
        logging.info(f"✅ FAISS memory loaded for org '{org_id}'")
    except Exception as e:
        logging.error(f"❌ Failed to load FAISS into memory for org '{org_id}'")


# ============================
# 🔨 14. BUILD FAISS INDEX FOR ONE ORG
# ============================

def build_faiss_index_from_training_dir(org_id):
    """
    Builds FAISS index for a specific org based on its training materials.
    Saves org-specific chunks, embeddings, and index.
    """
    paths = get_org_paths(org_id)
    training_dir = paths["training_data_dir"]
    all_chunks = []

    for root, _, files in os.walk(training_dir):
        for file in files:
            path = os.path.join(root, file)
            if file.endswith(".pdf"):
                raw = extract_text_from_pdf(path)
            elif file.endswith(".docx"):
                raw = extract_text_from_docx(path)
            elif file.lower().endswith((".png", ".jpg", ".jpeg")):
                raw = extract_text_from_image(path)
            else:
                continue

            chunks = chunk_text_by_words(raw)
            valid = [c for c in chunks if is_valid_chunk(c)]
            all_chunks.extend(valid)

    if not all_chunks:
        raise ValueError(f"❌ No valid chunks found for org '{org_id}'")

    # ✅ Encode and build FAISS
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    # ✅ Save outputs
    os.makedirs(paths["base"], exist_ok=True)
    with open(paths["chunks_pkl"], "wb") as f:
        pickle.dump(all_chunks, f)
    np.save(paths["embeddings_npy"], embeddings)
    faiss.write_index(index, paths["faiss_index"])

    logging.info(f"✅ FAISS index built for org '{org_id}' with {len(all_chunks)} chunks")

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
# 🌐 16. FASTAPI WEB SERVER FOR RUNPOD
# ============================

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import uvicorn

app = FastAPI(title="Athen.ai RAG API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class QueryRequest(BaseModel):
    question: str
    org_id: str
    k: int = 3

class QueryResponse(BaseModel):
    answer: str
    context_chunks: list
    org_id: str

class FineTuneRequest(BaseModel):
    org_id: str
    epochs: int = 3
    learning_rate: float = 2e-4
    batch_size: int = 1

class FineTuneResponse(BaseModel):
    message: str
    org_id: str
    status: str
    model_path: str

@app.get("/")
async def root():
    return {"message": "� Athen.ai RAG API is running on RunPod!"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    }

@app.post("/upload", response_model=dict)
async def upload_training_data(
    file: UploadFile = File(...),
    org_id: str = Form(...)
):
    """Upload ZIP file containing training materials for an organization"""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name
        
        # Process the upload
        handle_uploaded_zip(temp_path, org_id)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            "message": f"✅ Successfully uploaded and indexed training data for org '{org_id}'",
            "org_id": org_id,
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG system for a specific organization"""
    try:
        # Ensure org data is loaded
        if request.org_id not in ORG_FAISS_INDEXES:
            try:
                load_faiss_into_memory(request.org_id)
            except Exception:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No training data found for org '{request.org_id}'. Please upload training materials first."
                )
        
        # Retrieve context
        context_chunks = retrieve_context(
            query=request.question,
            k=request.k,
            org_id=request.org_id
        )
        
        # Generate answer
        answer = generate_rag_answer_with_context(
            user_question=request.question,
            context_chunks=context_chunks,
            mistral_tokenizer=tokenizer,
            mistral_model=rag_model
        )
        
        return QueryResponse(
            answer=answer,
            context_chunks=context_chunks,
            org_id=request.org_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/orgs")
async def list_organizations():
    """List all organizations with available models"""
    try:
        orgs = []
        if os.path.exists(ORG_DATA_ROOT):
            for org_dir in os.listdir(ORG_DATA_ROOT):
                org_path = os.path.join(ORG_DATA_ROOT, org_dir)
                if os.path.isdir(org_path):
                    paths = get_org_paths(org_dir)
                    has_index = os.path.exists(paths["faiss_index"])
                    has_model = os.path.exists(paths["model_dir"])
                    orgs.append({
                        "org_id": org_dir,
                        "has_training_data": has_index,
                        "has_fine_tuned_model": has_model,
                        "status": "ready" if has_index else "needs_training"
                    })
        
        return {"organizations": orgs}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list orgs: {str(e)}")

@app.post("/fine-tune", response_model=FineTuneResponse)
async def fine_tune_model(request: FineTuneRequest):
    """Fine-tune a model for a specific organization"""
    try:
        # Check if org has training data
        if request.org_id not in ORG_FAISS_INDEXES:
            try:
                load_faiss_into_memory(request.org_id)
            except Exception:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No training data found for org '{request.org_id}'. Please upload training materials first."
                )
        
        paths = get_org_paths(request.org_id)
        
        # Check if training data exists
        if not os.path.exists(paths["training_data_dir"]):
            raise HTTPException(
                status_code=404,
                detail=f"No training data directory found for org '{request.org_id}'"
            )
        
        # Create JSONL training file from training materials
        training_data = load_training_materials(paths["training_data_dir"])
        
        if training_data.empty:
            raise HTTPException(
                status_code=400,
                detail=f"No valid training data found for org '{request.org_id}'"
            )
        
        # Convert to JSONL format
        jsonl_path = os.path.join(paths["base"], "training_data.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for _, row in training_data.iterrows():
                json.dump({
                    "instruction": row["input"],
                    "output": row["output"]
                }, f)
                f.write("\n")
        
        # Create train/eval datasets
        train_dataset = MistralQADataset(jsonl_path, tokenizer, max_length=512)
        
        # Split for evaluation (use 10% for eval)
        eval_size = max(1, len(train_dataset) // 10)
        eval_indices = list(range(len(train_dataset)))[-eval_size:]
        train_indices = list(range(len(train_dataset)))[:-eval_size]
        
        from torch.utils.data import Subset
        train_subset = Subset(train_dataset, train_indices)
        eval_subset = Subset(train_dataset, eval_indices)
        
        # Fine-tune the model
        fine_tune_with_trainer(
            train_dataset=train_subset,
            eval_dataset=eval_subset,
            model=rag_model,
            tokenizer=tokenizer,
            output_dir=paths["model_dir"]
        )
        
        return FineTuneResponse(
            message=f"✅ Successfully fine-tuned model for org '{request.org_id}'",
            org_id=request.org_id,
            status="completed",
            model_path=paths["model_dir"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Fine-tuning failed: {str(e)}")

@app.post("/evaluate")
async def evaluate_model(org_id: str, questions: list[str] = None):
    """Evaluate the model for a specific organization"""
    try:
        # Default evaluation questions if none provided
        if not questions:
            questions = [
                "What are the key steps in laparoscopic appendectomy?",
                "Describe the blood supply to the pancreas",
                "What are the contraindications for robotic surgery?",
                "How do you manage postoperative bleeding after thyroidectomy?"
            ]
        
        # Check if org has training data
        if org_id not in ORG_FAISS_INDEXES:
            try:
                load_faiss_into_memory(org_id)
            except Exception:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No training data found for org '{org_id}'. Please upload training materials first."
                )
        
        paths = get_org_paths(org_id)
        eval_path = os.path.join(paths["base"], "evaluation_results.json")
        
        # Run evaluation
        evaluate_on_examples(
            model=rag_model,
            tokenizer=tokenizer,
            sample_questions=questions,
            save_path=eval_path,
            k=3,
            org_id=org_id
        )
        
        # Load and return results
        with open(eval_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        
        return {
            "org_id": org_id,
            "evaluation_results": results,
            "summary": {
                "total_questions": len(results),
                "avg_overlap_score": sum(r.get("overlap_score", 0) for r in results if r.get("overlap_score")) / len(results),
                "hallucination_count": sum(1 for r in results if r.get("hallucination_flag", False))
            }
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")

if __name__ == "__main__":
    # Load any existing org data on startup
    if os.path.exists(ORG_DATA_ROOT):
        for org_dir in os.listdir(ORG_DATA_ROOT):
            try:
                load_faiss_into_memory(org_dir)
                logging.info(f"✅ Loaded existing data for org '{org_dir}'")
            except Exception as e:
                logging.warning(f"⚠️ Could not load data for org '{org_dir}': {e}")
    
    # Start the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=19123)

# ============================
# 🌐 17. UPLOAD & INDEX MATERIALS (ORG-AWARE)
# ============================

import zipfile
import shutil

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
        load_faiss_into_memory(org_id)

        logging.info(f"🎉 Upload and indexing complete for org '{org_id}'")

    except Exception as e:
        logging.error(f"❌ Failed to handle ZIP upload for org '{org_id}': {e}")


