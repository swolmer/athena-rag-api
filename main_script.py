# ============================
# 🚀 1. IMPORTS & GLOBAL STORAGE
# ============================

# --- Standard library ---
import os
import json
import logging
import pickle
import argparse

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
# 🧠 GLOBAL ORG-SPECIFIC STORAGE
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
# 🛠️ 2. CONFIGURATION — COLLAB READY
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
# 📁 ORG PATH HELPER
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
# 2B. GLOBAL TOKENIZER & MODEL
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
# 3. GLOBAL EMBEDDING MODEL
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
# 4. UTILITIES
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
# 5. DATASET CLASS
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
# 6. TRAINING
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
# 7. RETRIEVAL FUNCTION — FIXED
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

    if not faiss_index or not rag_chunks or not rag_embeddings:
        raise ValueError(f"❌ FAISS data for org_id '{org_id}' not loaded.")

    try:
        query_embedding = embed_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
        D, I = faiss_index.search(query_embedding, initial_k)
        candidate_chunks = [rag_chunks[i] for i in I[0]]
        candidate_embeddings = [rag_embeddings[i] for i in I[0]]

        scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
        ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)

        return [chunk for chunk, _ in ranked[:k]]

    except Exception as e:
        logging.error(f"❌ Failed to retrieve context: {e}")
        return []


# ============================
# 8. RAG GENERATION — FIXED
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
# 9. EVALUATION FUNCTION — FIXED
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


def evaluate_on_examples(model, tokenizer, sample_questions, save_path="eval_outputs.json", k=3):
    """
    Evaluates the RAG chatbot on a list of questions using retrieved context and overlap scoring.
    Saves structured results to a JSON file.

    Parameters:
    - model: Hugging Face language model (Mistral or similar)
    - tokenizer: Corresponding tokenizer
    - sample_questions (list of str): Questions to evaluate
    - save_path (str): Where to store the evaluation output
    - k (int): Number of chunks to retrieve
    """
    global rag_model, faiss_index, rag_chunks, rag_embeddings  # required globals

    outputs = []

    for idx, question in enumerate(sample_questions, 1):
        print(f"\n🔹 Question {idx}/{len(sample_questions)}: {question}")

        try:
            # Step 1: Retrieve top-k chunks
            context_chunks = retrieve_context(question, k=k)
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
# 🔨 10A. BUILD FAISS INDEX FOR ONE ORG
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
# 📥 10B. LOAD FAISS FOR ONE ORG
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
# 10C. MAIN EXECUTION (CLI MODE)
# ============================
def main():
    global rag_model, faiss_index, rag_chunks, rag_embeddings
    rag_model = rag_model  # Already loaded globally
    faiss_index = None
    rag_chunks = []
    rag_embeddings = np.array([])

    print("🧠 Checking CUDA support:")
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
    print("🚀 Script has started running...")

    parser = argparse.ArgumentParser(description="Run RAG chatbot")
    parser.add_argument(
        '--jsonl_path',
        type=str,
        default=os.path.join(BASE_DIR, "step4_structured_instruction_finetune_ready.jsonl"),
        help="Path to JSONL training data"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=os.path.join(MODEL_DIR, "mistral-full-out"),
        help="Directory to save the model (if fine-tuning)"
    )
    args = parser.parse_args()

    model = rag_model  # ✅ Use global model
    load_rag_resources()

    sample_questions = [
        "How is a DIEP flap performed?",
        "What are the recommended closure techniques for the donor site following a DIEP flap procedure?",
        "What are the key anatomical landmarks and vascular considerations when injecting filler into the nasolabial folds?"
    ]

    evaluate_on_examples(
        model=model,
        tokenizer=tokenizer,
        sample_questions=sample_questions,
        save_path=os.path.join(BASE_DIR, "eval_outputs.json"),
        k=3
    )

    print("🩺 RAG Chatbot Ready. Ask your surgical question or type 'exit' to quit.")
    try:
        while True:
            user_question = input("You: ").strip()
            if user_question.lower() in {"exit", "quit"}:
                print("👋 Goodbye!")
                break
            if not user_question:
                continue

            context_chunks = retrieve_context(user_question)
            answer = generate_rag_answer_with_context(
                user_question=user_question,
                context_chunks=context_chunks,
                mistral_tokenizer=tokenizer,
                mistral_model=model
            )
            print("Bot:", answer)
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exiting chatbot.")


# ============================
# 10D. ENTRYPOINT
# ============================

if __name__ == "__main__":
    main()
