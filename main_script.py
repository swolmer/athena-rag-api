# ============================
# 1. IMPORTS
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
load_dotenv()  # ✅ Loads variables from .env file into environment

# ✅ Read API key from .env (used in FastAPI server or secure client access)
RAG_API_KEY = os.getenv("RAG_API_KEY", "")

# --- Natural Language Tokenization ---
import nltk
import nltk.data
from nltk.tokenize.punkt import PunktSentenceTokenizer

# ✅ Ensure NLTK punkt tokenizer is downloaded to a persistent directory
nltk_data_path = os.path.join(os.path.expanduser("~"), "nltk_data")
nltk.download("punkt", download_dir=nltk_data_path)
nltk.data.path.append(nltk_data_path)

# ✅ Safe sentence tokenizer to avoid NLTK punkt_tab bug
def safe_sent_tokenize(text, lang='english'):
    try:
        # Load pretrained Punkt tokenizer from NLTK's path
        punkt_path = nltk.data.find(f'tokenizers/punkt/{lang}.pickle')
        with open(punkt_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer.tokenize(text)
    except Exception as e:
        print(f"❌ Custom sent_tokenize failed: {e}")
        return text.split('.')  # Fallback


# Hugging Face model/tokenizer/trainer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoConfig,
    Trainer,
    EarlyStoppingCallback,
    TrainingArguments
)

# CUDA device info
print("🧠 Checking CUDA support:")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ No CUDA-compatible GPU detected. Training will run on CPU.")

# File extraction utilities
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from docx import Document

# RAG embedding + retrieval
from sentence_transformers import SentenceTransformer
import faiss

# Evaluation utilities
from evaluate import load as load_metric

# Optional Hugging Face token login
# from huggingface_hub import login  # Only needed if pushing to hub or accessing gated models

# ============================
# 2. CONFIGURATION — FIXED
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index", "faiss.index")
CHUNKS_PKL_PATH = os.path.join(BASE_DIR, "faiss_index", "rag_chunks.pkl")
EMBEDDINGS_NPY_PATH = os.path.join(BASE_DIR, "faiss_index", "rag_embeddings.npy")
LLM_MODEL_NAME = "mistralai/Mistral-7B-v0.1"  # or use a smaller open LLM

# ✅ Optional: Load Hugging Face token (for gated/private models or pushing to hub)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if HF_TOKEN:
    from huggingface_hub import login
    login(token=HF_TOKEN)

# ✅ Setup logging
logging.basicConfig(level=logging.INFO)

# ✅ Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🧠 Checking CUDA support:")
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA Device Name:", torch.cuda.get_device_name(0))
else:
    print("⚠️ CUDA not available — using CPU")

# ✅ Base working directory
BASE_DIR = r"C:\Users\sophi\Downloads\MyFlaskApp"

# ✅ Directory structure
MODEL_DIR = os.path.join(BASE_DIR, "Models")
TRAINING_DATA_DIR = os.path.join(BASE_DIR, "training set")
CSV_PATH = os.path.join(BASE_DIR, "Training_QA_Pairs.csv")

# ✅ RAG-specific paths
FAISS_INDEX_PATH = os.path.join(BASE_DIR, "faiss_index.idx")
CHUNKS_PKL_PATH = os.path.join(BASE_DIR, "rag_chunks.pkl")
EMBEDDINGS_NPY_PATH = os.path.join(BASE_DIR, "rag_embeddings.npy")

# ✅ Model identifiers
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
LLM_MODEL_NAME = "NousResearch/Hermes-2-Pro-Mistral-7B"

# ✅ Load embedding model globally
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embed_model = embed_model.to(DEVICE)
    globals()["embed_model"] = embed_model
    logging.info(f"✅ Loaded embedding model '{EMBEDDING_MODEL_NAME}' on {DEVICE}")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model: {e}")
    embed_model = None


# ============================
# 3. EMBEDDING MODEL LOAD
# ============================

from sentence_transformers import SentenceTransformer

# Set device for embedding model (defaults to CUDA if available)
EMBED_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the embedding model globally for context retrieval and FAISS
try:
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=EMBED_DEVICE)
    globals()["embed_model"] = embed_model  # Ensure accessible across functions
    logging.info(f"✅ Loaded embedding model '{EMBEDDING_MODEL_NAME}' on device: {EMBED_DEVICE}")
except Exception as e:
    logging.error(f"❌ Failed to load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
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

from sklearn.metrics.pairwise import cosine_similarity

# Load these earlier in your script or notebook:
# embed_model = SentenceTransformer("all-MiniLM-L6-v2")
# rag_chunks = pickle.load(open("rag_chunks.pkl", "rb"))
# rag_embeddings = np.load("rag_embeddings.npy")
# faiss_index = faiss.read_index("faiss_index.idx")

def retrieve_context(query, k=3, initial_k=10):
    """
    Retrieves the k most relevant chunks using FAISS followed by cosine similarity re-ranking.

    Parameters:
    - query (str): User's question
    - k (int): Number of top-ranked chunks to return
    - initial_k (int): Number of initial candidates retrieved from FAISS

    Returns:
    - List[str]: Top-k re-ranked context chunks most relevant to the query
    """
    if not query or not hasattr(embed_model, 'encode'):
        raise ValueError("❌ Query is empty or embed_model not initialized")

    try:
        # Step 1: Embed query
        query_embedding = embed_model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)

        # Step 2: FAISS search
        D, I = faiss_index.search(query_embedding, initial_k)
        candidate_chunks = [rag_chunks[i] for i in I[0]]
        candidate_embeddings = [rag_embeddings[i] for i in I[0]]

        # Step 3: Re-rank with cosine similarity
        scores = cosine_similarity(query_embedding, candidate_embeddings)[0]
        ranked = sorted(zip(candidate_chunks, scores), key=lambda x: x[1], reverse=True)

        # Step 4: Return top-k chunks
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
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.1,
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
# 10A. BUILD FAISS INDEX
# ============================

def build_faiss_index_from_training_dir(training_dir):
    """
    Walks through training materials and builds a FAISS index from valid text chunks.
    Saves chunks, embeddings, and FAISS index to disk.
    """
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

    embeddings = embed_model.encode(all_chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    with open(CHUNKS_PKL_PATH, "wb") as f:
        pickle.dump(all_chunks, f)
    np.save(EMBEDDINGS_NPY_PATH, embeddings)
    faiss.write_index(index, FAISS_INDEX_PATH)

    print(f"✅ Built FAISS index with {len(all_chunks)} chunks")
# ============================
# 10. MAIN EXECUTION — CLEAN GPU VERSION
# ============================

if __name__ == "__main__":
    global rag_model, faiss_index, rag_chunks, rag_embeddings
    rag_model = None
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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load full-precision model (no quantization, no LoRA)
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # 🔧 Build FAISS index if missing
    if not os.path.exists(FAISS_INDEX_PATH) or not os.path.exists(CHUNKS_PKL_PATH) or not os.path.exists(EMBEDDINGS_NPY_PATH):
        print("⚙️ FAISS resources not found. Building from training materials...")
        build_faiss_index_from_training_dir(TRAINING_DATA_DIR)

    # Load RAG resources
    try:
        with open(CHUNKS_PKL_PATH, "rb") as f:
            rag_chunks = pickle.load(f)
        rag_embeddings = np.load(EMBEDDINGS_NPY_PATH)
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        logging.info(f"✅ Loaded FAISS index and {len(rag_chunks)} chunks")
    except Exception as e:
        logging.error(f"❌ Failed to load FAISS resources: {e}")
        rag_chunks = []
        rag_embeddings = np.array([])
        faiss_index = None

    rag_model = model  # use the model for answering

    # Sample questions
    sample_questions = [
        "How is a DIEP flap performed?",
        "What are the recommended closure techniques for the donor site following a DIEP flap procedure?",
        "What are the key anatomical landmarks and vascular considerations when injecting filler into the nasolabial folds?"
    ]

    # Evaluate
    evaluate_on_examples(
        model=model,
        tokenizer=tokenizer,
        sample_questions=sample_questions,
        save_path=os.path.join(BASE_DIR, "eval_outputs.json"),
        k=3
    )

# ✅ Launch interactive chatbot REPL
print("🩺 RAG Chatbot Ready. Ask your surgical question or type 'exit' to quit.")
try:
    while True:
        user_question = input("You: ").strip()
        if user_question.lower() in {"exit", "quit"}:
            print("👋 Goodbye!")
            break
        if not user_question:
            continue  # Skip empty inputs

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

    # Launch chatbot
    print("🩺 RAG Chatbot Ready. Ask your surgical question or type 'exit' to quit.")
    try:
        while True:
            user_question = input("You: ")
            if user_question.strip().lower() in {"exit", "quit"}:
                break
            context_chunks = retrieve_context(user_question)
            answer = generate_rag_answer_with_context(user_question, context_chunks, tokenizer, model)
            print("Bot:", answer)
    except (KeyboardInterrupt, EOFError):
        print("\n👋 Exiting chatbot. Goodbye.")
