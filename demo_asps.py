# ============================
# �️ MISSING IMPORTS & GLOBALS
# ============================
import os, sys, json, logging, pickle, numpy as np, shutil, zipfile, argparse, concurrent.futures
import torch
import faiss
import fitz
import pytesseract
import urllib.request
import pandas as pd
# ============================
# �🚀 1. IMPORTS & GLOBAL STORAGE
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
async def chatbot_ui():
    # Serve the HTML UI from a file for easier editing and proper rendering
    html_path = os.path.join(os.path.dirname(__file__), "attached_assets", "chatbot_ui.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<h2>UI file not found or error loading: {e}</h2>", status_code=500)
# --- Training-related code commented out for deployment ---

from transformers import TrainingArguments, Trainer, default_data_collator
import torch

def train_model(model, tokenizer, train_dataset, eval_dataset, output_dir, debug=False):
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=default_data_collator
    )

    print("🚀 Starting training...")
    trainer.train()

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
    # Fallback: If clinical context is empty, try navigation context
    if not context_chunks and intent == "clinical":
        print("⚠️ No clinical context found, falling back to navigation context...")
        fallback_chunks = retrieve_context(user_question, k=3, org_id=None, intent="navigation")
        if fallback_chunks:
            context_chunks = fallback_chunks
            intent = "navigation"  # Switch format to navigation
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
                   "🩺 **Consult a board-certified plastic surgeon** - They can provide personalized, evidence-based advice\n"
                   "📚 **Review peer-reviewed literature** - Look for recent studies on your specific concern\n"
                   "🏥 **Speak with your healthcare provider** - They know your medical history and current health status\n"
                   "📞 **Contact ASPS** at (847) 228-9900 for surgeon referrals in your area\n\n"
                   "Your health and safety are paramount - professional medical consultation is always the safest approach for clinical questions.")

    context = "\n\n".join(f"- {chunk.strip()}" for chunk in context_chunks)
    # Different formats based on question intent
    if intent == "navigation":
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
        # Clinical format for medical/surgical questions with improved emoji formatting
        prompt = (
            "You are a surgical expert writing answers for a clinical reference guide.\n"
            "Use only the CONTEXT below to answer the QUESTION in this clinical format. Make the answer visually appealing with emojis and clear sections:\n\n"
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
            max_new_tokens=256,     # increased length for fuller answers
            do_sample=False,        # deterministic, faster
            temperature=0.1,        # low temperature for more focused responses
            repetition_penalty=1.2, # prevent repetitive text
            eos_token_id=mistral_tokenizer.eos_token_id,
            pad_token_id=mistral_tokenizer.pad_token_id,
            early_stopping=True     # stop at natural ending points
        )

    decoded = mistral_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract portion after '### ANSWER:'
    if "### ANSWER:" in decoded:
        answer = decoded.split("### ANSWER:")[-1].strip()
    else:
        answer = decoded.strip()
    # Clean up repeated characters and invalid sequences
    import re
    answer = re.sub(r'\b(\d)\s+\1(\s+\1)+', '', answer)  # Remove repeated digits with spaces
    answer = re.sub(r'(\w)\1{3,}', r'\1', answer)        # Remove excessive character repetition
    answer = re.sub(r'\s+', ' ', answer)                 # Normalize whitespace
    answer = re.sub(r'\b(the the|and and|of of|in in)\b', r'\1'.split()[0], answer)  # Remove repeated words
    answer = re.sub(r'[^\w\s\.,!?:;()-]', '', answer)    # Remove invalid characters

    # Pretty formatting for clinical answers with emojis and sections
    if intent == "clinical":
        summary = anatomy = procedure = pearls = ""
        summary_match = re.search(r'✅ Summary:(.*?)(🧠|🔧|⚠️|$)', answer, re.DOTALL)
        anatomy_match = re.search(r'🧠 Anatomy & Physiology:(.*?)(🔧|⚠️|$)', answer, re.DOTALL)
        procedure_match = re.search(r'🔧 Procedure or Technique:(.*?)(⚠️|$)', answer, re.DOTALL)
        pearls_match = re.search(r'⚠️ Pitfalls & Pearls:(.*)', answer, re.DOTALL)
        if summary_match:
            summary = summary_match.group(1).strip()
        if anatomy_match:
            anatomy = anatomy_match.group(1).strip()
        if procedure_match:
            procedure = procedure_match.group(1).strip()
        if pearls_match:
            pearls = pearls_match.group(1).strip()
        pretty_answer = f"✅ Summary: {summary}\n\n🧠 Anatomy & Physiology: {anatomy}\n\n🔧 Procedure or Technique: {procedure}\n\n⚠️ Pitfalls & Pearls: {pearls}"
        answer = pretty_answer.strip()

    sentences = re.split(r'(?<=[.!?])\s+', answer)
    if len(sentences) > 8:
        answer = " ".join(sentences[:8]).strip()
    else:
        answer = answer.strip()

    if not answer.endswith(('.', '!', '?')):
        answer += "."

    answer_tokens = set(re.findall(r"\b\w+\b", answer.lower()))
    context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
    overlap = answer_tokens & context_tokens
    overlap_score = len(overlap) / max(1, len(answer_tokens))
    print(f"🔍 Hallucination check - Overlap score: {overlap_score:.2f}")
    min_threshold = 0.25
    if overlap_score < min_threshold:
        logging.warning(f"⚠️ Very low token overlap ({overlap_score:.2f}) — likely hallucination.")
        if intent == "navigation":
            return ("I don't have enough reliable information to provide accurate guidance about this specific website navigation question. "
                   "Rather than potentially mislead you, I recommend getting current information directly from:\n\n"
                   "📍 **plasticsurgery.org** - Visit the official site for the most up-to-date features\n"
                   "🔍 **Site search** - Use their search function for specific topics\n"
                   "📞 **ASPS support** - Call (847) 228-9900 for personalized website assistance\n"
                   "💬 **Live chat** - Check if they offer live support for navigation questions\n\n"
                   "This ensures you get accurate, current information about their website resources and tools.")
        else:
            return ("I cannot provide confident medical information for this specific clinical question, as I prioritize accuracy and your safety above all else. "
                   "Rather than risk giving you potentially incorrect medical guidance that could affect your health decisions, I strongly recommend:\n\n"
                   "🩺 **Board-certified plastic surgeon consultation** - Get personalized, professional medical advice\n"
                   "📚 **Peer-reviewed medical literature** - Research current, evidence-based studies on your topic\n"
                   "🏥 **Your healthcare provider** - They understand your medical history and current health status\n"
                   "📞 **ASPS surgeon referral** - Call (847) 228-9900 to find qualified specialists in your area\n"
                   "🌐 **ASPS patient education resources** - Visit plasticsurgery.org for verified patient information\n\n"
                   "Your health and safety are my top priorities - professional medical consultation is always the most reliable path for clinical guidance.")
    else:
        print(f"✅ Answer passed hallucination check with {overlap_score:.2f} overlap")
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
            hallucinated = overlap_score < 0.25  # Use same improved threshold

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
    
    # Define all clinical training directories that should be in GitHub repo
    clinical_dirs_github = [
        "Training Data Op",
        "Training Data Textbooks", 
        "Validate",
        "op notes",
        "textbook notes",
        "clinical"
    ]
    
    # Also check local paths (fallback for development)
    clinical_dirs_local = [
        ("clinical", os.path.join(BASE_DIR, "clinical")),
        ("Training Data Op", os.path.join(BASE_DIR, "Training Data Op")),
        ("Training Data Textbooks", os.path.join(BASE_DIR, "Training Data Textbooks")), 
        ("Validate", os.path.join(BASE_DIR, "Validate")),
        ("op notes", os.path.join(BASE_DIR, "op notes")),
        ("textbook notes", os.path.join(BASE_DIR, "textbook notes"))
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
    github_branch = "asps_demo"  # Use the correct branch where files are located
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
    
    print("📄 JSON Knowledge Base Files (Root Directory):")
    print("   ✅ navigation_training_data.json   - Original navigation training data")
    print("   ✅ nav1.json                       - Clinical content (20.88MB, 31,893 chunks)")
    print("   ✅ nav2.json                       - Navigation content (17.28MB, 14,649 chunks)")
    print("   📦 ultimate_asps_knowledge_base.json - Full backup file (37.37MB)")
    print("")
    
    print("📁 Clinical Training Directories (Should be in repository):")
    print("   ✅ Training Data Op/               - Operative procedure PDFs/DOCX")
    print("   ✅ Training Data Textbooks/        - Medical textbook materials")
    print("   ✅ Validate/                       - Validation datasets")
    print("   ✅ op notes/                       - Operative notes")
    print("   ✅ textbook notes/                 - Textbook note summaries")
    print("   ✅ clinical/                       - General clinical materials")
    print("")
    
    print("🔧 RUNPOD DEPLOYMENT PROCESS:")
    print("   1️⃣ Provide GitHub token in RunPod environment")
    print("   2️⃣ Clone repository: git clone https://github.com/swolmer/athena-rag-api.git")
    print("   3️⃣ Run: python demo_asps.py")
    print("   4️⃣ System will automatically:")
    print("      - Download nav1.json, nav2.json, navigation_training_data.json")
    print("      - Load clinical training directories from cloned repo")
    print("      - Build dual FAISS indexes (clinical + navigation)")
    print("      - Start training/inference")
    print("")
    
    print("💡 INDEX MAPPING:")
    print("   📚 CLINICAL INDEX uses:")
    print("      - nav1.json (clinical content from knowledge base)")
    print("      - Training Data Op/ (operative procedures)")
    print("      - Training Data Textbooks/ (medical textbooks)")
    print("      - Validate/, op notes/, textbook notes/, clinical/")
    print("")
    print("   🧭 NAVIGATION INDEX uses:")
    print("      - nav2.json (navigation content from knowledge base)")
    print("      - navigation_training_data.json (original navigation data)")
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
    
    Clinical: Uses original training materials (PDFs, medical content)
    Navigation: Uses scraped ASPS website content for site navigation
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

    # Ensure scraped_content_dir is defined before use
    scraped_content_dir = os.path.join(paths["base"], "extracted_content")

    if not os.path.exists(scraped_content_dir):
        os.makedirs(scraped_content_dir, exist_ok=True)
        print(f"📁 Created missing directory: {scraped_content_dir}")
        print("⚠️ No navigation JSON files found yet. Please upload your files to this directory and rerun the script.")
        navigation_chunks = []
    else:
        print(f"✅ Found content directory: {scraped_content_dir}")
        navigation_jsons = []
        for fname in os.listdir(scraped_content_dir):
            if fname.lower().endswith(".json"):
                fpath = os.path.join(scraped_content_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Accept either a list of chunks or a single content string
                        if isinstance(data, dict) and "content" in data:
                            navigation_chunks.append(data["content"])
                        elif isinstance(data, list):
                            navigation_chunks.extend(data)
                        else:
                            print(f"⚠️ Unrecognized JSON format in {fname}")
                        navigation_jsons.append(fname)
                except Exception as e:
                    print(f"❌ Error loading {fname}: {e}")
        print(f"✅ Loaded {len(navigation_jsons)} navigation JSON files from extracted_content.")
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
    - nav1.json: Clinical content split (20.88 MB, 31,893 chunks)
    - nav2.json: Navigation content split (17.28 MB, 14,649 chunks)
    
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
        "nav1.json",                      # 🆕 Clinical content split (20.88 MB, 31,893 chunks) → CLINICAL INDEX
        "nav2.json",                      # 🆕 Navigation content split (17.28 MB, 14,649 chunks) → NAVIGATION INDEX
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
        
        # Try to load different knowledge base files (from repo root)
        kb_files = [
            ("navigation_training_data.json", "navigation"),      # Original navigation training data
            ("nav1.json", "clinical"),                           # Clinical content split (formerly clinical_knowledge_base.json)
            ("nav2.json", "navigation"),                         # Navigation content split (formerly navigation_knowledge_base.json) 
            ("ultimate_asps_knowledge_base.json", "mixed")       # Full comprehensive file as fallback
        ]
        
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
        # 📚 STEP 2: LOAD CLINICAL TRAINING DIRECTORIES (NEW!)
        # ============================
        print("📚 Loading clinical training directories for CLINICAL FAISS...")
        
        # Define all clinical training directories that should be loaded
        clinical_training_dirs = [
            "Training Data Op",
            "Training Data Textbooks", 
            "Validate",
            "op notes",
            "textbook notes",
            "clinical"
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
        print(f"   📚 Clinical chunks: {len(clinical_chunks)} (JSON + directories)")
        print(f"   🧭 Navigation chunks: {len(navigation_chunks)} (JSON only)")
        
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
        
    except Exception as e:
        print(f"❌ Failed to load GitHub knowledge bases: {e}")
        traceback.print_exc()
        return False

# ============================
# 🌐 FASTAPI DEMO ENDPOINTS
# ============================


# --- FastAPI and dependencies ---
import os
import traceback
import shutil
import pickle
import logging
import json
import urllib.request
import urllib.error
import numpy as np
import torch
import argparse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

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
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ASPS Medical AI Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0; padding: 0;
            min-height: 100vh;
            background: linear-gradient(135deg, #7b2ff2 0%, #f357a8 100%);
        }
        .main-layout {
            display: flex;
            min-height: 100vh;
        }
        .sidebar {
            width: 260px;
            background: rgba(255,255,255,0.12);
            color: #fff;
            display: flex;
            flex-direction: column;
            padding: 32px 16px 16px 16px;
            box-shadow: 2px 0 12px rgba(123,47,242,0.08);
        }
        .sidebar h2 {
            font-size: 1.3em;
            margin-bottom: 18px;
            color: #fff;
            letter-spacing: 1px;
        }
        .sidebar .quick-actions {
            margin-bottom: 24px;
        }
        .sidebar button {
            display: block;
            width: 100%;
            margin-bottom: 10px;
            background: #fff;
            color: #7b2ff2;
            border: none;
            border-radius: 6px;
            padding: 10px 0;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.2s;
        }
        .sidebar button:hover {
            background: #f357a8;
            color: #fff;
        }
        .sidebar .status {
            margin-top: auto;
            font-size: 0.98em;
            background: rgba(255,255,255,0.18);
            border-radius: 6px;
            padding: 10px;
            color: #fff;
        }
        .container {
            flex: 1;
            max-width: 700px;
            margin: 40px auto;
            background: #fff;
            border-radius: 12px;
            box-shadow: 0 2px 16px rgba(123,47,242,0.10);
            padding: 32px 40px 24px 40px;
            display: flex;
            flex-direction: column;
        }
        .branding {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 18px;
        }
        .branding img {
            height: 38px;
        }
        h1 {
            color: #7b2ff2;
            font-size: 2em;
            margin-bottom: 8px;
        }
        .chatbox {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 18px;
            background: #f9f9fc;
            min-height: 220px;
            margin-bottom: 18px;
            overflow-y: auto;
            max-height: 340px;
        }
        .user { color: #7b2ff2; font-weight: bold; margin-bottom: 6px; }
        .bot { color: #1b8e5a; font-weight: bold; margin-bottom: 6px; }
        .input-row {
            display: flex;
            gap: 10px;
            margin-bottom: 8px;
        }
        input[type="text"] {
            flex: 1;
            padding: 12px;
            border-radius: 6px;
            border: 1px solid #ccc;
            font-size: 1.08em;
        }
        button.send-btn {
            background: linear-gradient(90deg, #7b2ff2 0%, #f357a8 100%);
            color: #fff;
            border: none;
            border-radius: 6px;
            padding: 12px 22px;
            cursor: pointer;
            font-weight: bold;
            font-size: 1em;
        }
        button.send-btn:hover {
            background: #1b8e5a;
        }
        .footer {
            text-align: center;
            color: #888;
            margin-top: 18px;
            font-size: 0.98em;
        }
        .disclaimer {
            background: #f9f9fc;
            border-radius: 6px;
            padding: 10px;
            color: #7b2ff2;
            font-size: 0.97em;
            margin-bottom: 10px;
        }
        .loading {
            color: #f357a8;
            font-style: italic;
            margin-bottom: 8px;
        }
        @media (max-width: 900px) {
            .main-layout { flex-direction: column; }
            .sidebar { width: 100%; min-height: 0; box-shadow: none; }
            .container { max-width: 100%; padding: 18px; }
        }
    </style>
</head>
<body>
    <div class="main-layout">
        <div class="sidebar">
            <h2>Clinical Topics</h2>
            <div class="quick-actions">
                <button onclick="quickAsk('What are the typical indications for placement of a tissue expander in breast reconstruction surgery?')">Tissue Expander Indications</button>
                <button onclick="quickAsk('Explain the operative steps for a free TRAM flap breast reconstruction.')">TRAM Flap Steps</button>
                <button onclick="quickAsk('What is the vascular supply of the radial forearm flap?')">Radial Forearm Flap</button>
                <button onclick="quickAsk('How is capsulorrhaphy performed during implant exchange in breast reconstruction?')">Capsulorrhaphy</button>
                <button onclick="quickAsk('What precautions must be taken to avoid injury to the peroneal nerve during fibula flap harvest?')">Fibula Flap Precautions</button>
            </div>
            <h2>Navigation</h2>
            <div class="quick-actions">
                <button onclick="quickAsk('Where can I find information about breast augmentation costs?')">Breast Augmentation Costs</button>
                <button onclick="quickAsk('How do I locate a plastic surgeon in my area?')">Find a Surgeon</button>
                <button onclick="quickAsk('Where are the before and after photos on the ASPS website?')">Photo Gallery</button>
                <button onclick="quickAsk('How do I navigate to patient safety information?')">Patient Safety Info</button>
            </div>
            <div class="status" id="statusBox">
                <strong>Connection:</strong> <span id="connStatus">Checking...</span><br>
                <strong>GPU:</strong> <span id="gpuStatus">-</span><br>
                <strong>Model:</strong> <span id="modelStatus">-</span>
            </div>
        </div>
        <div class="container">
            <div class="branding">
                <img src="https://www.plasticsurgery.org/images/logo.svg" alt="ASPS Logo" />
                <h1>ASPS Medical Assistant</h1>
            </div>
            <div class="disclaimer">
                <strong>Disclaimer:</strong> This chatbot provides medical information for educational purposes only. Always consult a qualified healthcare provider for medical advice.
            </div>
            <div class="chatbox" id="chatbox">
                <div class="bot">🤖 Welcome! Ask a clinical or navigation question about plastic surgery.</div>
            </div>
            <div class="input-row">
                <input type="text" id="userInput" placeholder="Type your question here..." autofocus />
                <button class="send-btn" onclick="sendMessage()">Send</button>
            </div>
            <div class="loading" id="loadingBox" style="display:none;">AI is thinking<span id="dots">...</span></div>
            <div class="footer">
                Powered by Athena RAG API &mdash; For demonstration purposes only.<br>
                <span id="responseTime"></span>
            </div>
        </div>
    </div>
    <script>
        const chatbox = document.getElementById('chatbox');
        const userInput = document.getElementById('userInput');
        const loadingBox = document.getElementById('loadingBox');
        const dots = document.getElementById('dots');
        const connStatus = document.getElementById('connStatus');
        const gpuStatus = document.getElementById('gpuStatus');
        const modelStatus = document.getElementById('modelStatus');
        const responseTime = document.getElementById('responseTime');

        function appendMessage(sender, text) {
            const div = document.createElement('div');
            div.className = sender;
            div.textContent = (sender === 'user' ? '🧑 ' : '🤖 ') + text;
            chatbox.appendChild(div);
            chatbox.scrollTop = chatbox.scrollHeight;
        }

        function quickAsk(question) {
            userInput.value = question;
            sendMessage();
        }

        let loadingInterval;
        function showLoading() {
            loadingBox.style.display = '';
            let dotCount = 0;
            loadingInterval = setInterval(() => {
                dotCount = (dotCount + 1) % 4;
                dots.textContent = '.'.repeat(dotCount + 1);
            }, 400);
        }
        function hideLoading() {
            loadingBox.style.display = 'none';
            clearInterval(loadingInterval);
            dots.textContent = '...';
        }

        async function sendMessage() {
            const question = userInput.value.trim();
            if (!question) return;
            appendMessage('user', question);
            userInput.value = '';
            showLoading();
            const start = performance.now();
            try {
                const res = await fetch('/query', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question, k: 3 })
                });
                const data = await res.json();
                hideLoading();
                const end = performance.now();
                responseTime.textContent = `Response time: ${(end - start).toFixed(1)} ms`;
                if (data.answer) {
                    appendMessage('bot', data.answer);
                } else {
                    appendMessage('bot', 'No answer received.');
                }
            } catch (err) {
                hideLoading();
                appendMessage('bot', 'Error: ' + err);
            }
        }

        userInput.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') sendMessage();
        });

        // Health monitoring
        async function updateStatus() {
            try {
                const res = await fetch('/health');
                const data = await res.json();
                connStatus.textContent = data.status === 'healthy' ? 'Connected' : 'Initializing...';
                gpuStatus.textContent = data.gpu_name || '-';
                modelStatus.textContent = data.models_loaded && data.models_loaded.rag_model ? 'Loaded' : 'Loading...';
            } catch {
                connStatus.textContent = 'Offline';
                gpuStatus.textContent = '-';
                modelStatus.textContent = '-';
            }
        }
        updateStatus();
        setInterval(updateStatus, 15000);
    </script>
</body>
</html>
'''
    return HTMLResponse(content=html_content)

@app.get("/api")
async def api_info():
    return {"message": "🩺 ASPS RAG Demo API is running!", "status": "ready"}

@app.get("/health")
async def health_check():
    """Enhanced health check with detailed system status"""
    try:
        # Check FAISS indexes status
        asps_loaded = "asps" in ORG_FAISS_INDEXES
        clinical_count = 0
        navigation_count = 0
        
        if asps_loaded:
            clinical_chunks = ORG_CHUNKS.get("asps", {}).get("clinical", [])
            navigation_chunks = ORG_CHUNKS.get("asps", {}).get("navigation", [])
            clinical_count = len(clinical_chunks)
            navigation_count = len(navigation_chunks)
        
        # Check model status
        models_loaded = {
            "tokenizer": tokenizer is not None,
            "rag_model": rag_model is not None, 
            "embed_model": embed_model is not None
        }
        
        return {
            "status": "healthy" if asps_loaded else "initializing",
            "timestamp": str(torch.tensor(0).device),  # Simple way to get current timestamp
            
            # Hardware info
            "cuda_available": torch.cuda.is_available(),
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None",
            
            # Data status  
            "org_loaded": asps_loaded,
            "clinical_chunks_count": clinical_count,
            "navigation_chunks_count": navigation_count,
            "total_chunks": clinical_count + navigation_count,
            
            # Model status
            "models_loaded": models_loaded,
            "all_models_ready": all(models_loaded.values()),
            
            # System readiness
            "system_ready": asps_loaded and all(models_loaded.values()),
            
            # API info
            "endpoints": ["/", "/api", "/health", "/query", "/api/chat", "/sample-questions"],
            "port": int(os.environ.get("PORT", 19524))
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "cuda_available": torch.cuda.is_available() if 'torch' in globals() else False,
            "system_ready": False
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
    return {
        "questions": [
            "Describe the steps involved in the placement of a tissue expander after mastectomy.",
            "What is the vascular supply of the radial forearm flap?",
            "What precautions must be taken to avoid injury to the peroneal nerve during fibula flap harvest?",
            "What are the key elements of patient positioning and prep for a TRAM flap procedure?",
            "How is capsulorrhaphy performed during implant exchange in breast reconstruction?",
            "What are the differences between craniofacial and mandibular plates?",
            # ...add more from your tailored list...
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
                print("❌ Failed to load knowledge bases into memory")
                print("🔄 Falling back to local content and building basic indexes...")
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
                    print("✅ Built new indexes:")
                    print(f"   📚 Clinical chunks: {result['clinical_chunks']}")
                    print(f"   🧭 Navigation chunks: {result['navigation_chunks']}")
    except Exception as e:
        print(f"❌ Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()

# ============================
# 🛠️ MISSING STUBS FOR UNDEFINED SYMBOLS
# ============================
def get_org_paths(org_id):
    """Returns a dict of paths for the given org_id."""
    base = os.path.join(BASE_DIR, "org_data", org_id)
    return {
        "base": base,
        "clinical_training_dir": os.path.join(base, "clinical"),
        "training_data_dir": os.path.join(base, "training_data"),
        "chunks_pkl": os.path.join(base, "clinical_chunks.pkl"),
        "embeddings_npy": os.path.join(base, "clinical_embeddings.npy"),
        "faiss_index": os.path.join(base, "clinical_index.faiss"),
    }

def is_valid_chunk(chunk):
    """Returns True if chunk is valid (not empty, not too short)."""
    return isinstance(chunk, str) and len(chunk.strip()) > 30

def cosine_similarity(a, b):
    """Returns cosine similarity between two numpy arrays."""
    a = np.array(a)
    b = np.array(b)
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)

# ============================
# 🛠️ MISSING IMPORTS & GLOBALS
# ============================
import os, sys, json, logging, pickle, numpy as np, shutil, zipfile, argparse, concurrent.futures
import torch
import faiss
import fitz
import pytesseract
import urllib.request
import pandas as pd

# ============================
# 🛠️ GLOBAL VARIABLES
# ============================
ORG_FAISS_INDEXES = {}
ORG_CHUNKS = {}
ORG_EMBEDDINGS = {}
rag_model = None
faiss_index = None
rag_chunks = None
rag_embeddings = None
tokenizer = None
embed_model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================
# 🛠️ UTILITY FUNCTION STUBS
# ============================
def chunk_text_by_words(text, max_words=800):
    """Splits text into chunks of max_words words."""
    if not text:
        return []
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def classify_question_intent(question):
    """Classifies question as 'clinical' or 'navigation'."""
    clinical_keywords = ["surgery", "operative", "flap", "procedure", "blood supply", "implant", "reconstruction", "harvest", "preoperative", "muscle", "plate", "fibula", "capsulorrhaphy", "tissue expander"]
    navigation_keywords = ["find", "locate", "where", "navigate", "cost", "photo", "gallery", "safety", "recovery", "tool", "information"]
    q = question.lower()
    if any(kw in q for kw in clinical_keywords):
        return "clinical"
    if any(kw in q for kw in navigation_keywords):
        return "navigation"
    return "clinical"  # Default fallback

def retrieve_context(query, k=3, org_id="asps", intent="clinical"):
    """Retrieves top-k context chunks from FAISS index."""
    if org_id not in ORG_FAISS_INDEXES:
        return []
    index = ORG_FAISS_INDEXES[org_id][intent]
    chunks = ORG_CHUNKS[org_id][intent]
    embeddings = ORG_EMBEDDINGS[org_id][intent]
    # Use embed_model to encode query
    if embed_model is None:
        return []
    query_emb = embed_model.encode([query])
    D, I = index.search(np.array(query_emb), k)
    return [chunks[i] for i in I[0] if i < len(chunks)]

def generate_rag_answer_with_context(user_question, context_chunks, mistral_tokenizer, mistral_model, intent="clinical"):
    """Generates answer using context and model."""
    # This is a stub; replace with actual model inference
    context = "\n".join(context_chunks)
    return f"[Context: {intent}] {context}\nAnswer: This is a generated answer for '{user_question}'."

# ============================
# 🛠️ MODEL LOADING STUBS
# ============================
def load_models():
    global rag_model, tokenizer, embed_model
    # Replace with actual model loading code
    rag_model = "Loaded_Mistral_Model"
    tokenizer = "Loaded_Mistral_Tokenizer"
    class DummyEmbedModel:
        def encode(self, texts, show_progress_bar=False):
            # Return random vectors for demo
            return np.random.rand(len(texts), 768)
    embed_model = DummyEmbedModel()
    print("✅ Models loaded (stub)")

load_models()

