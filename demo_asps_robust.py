#!/usr/bin/env python3
"""
🏥 ASPS MEDICAL AI CHATBOT - ROBUST DUAL KNOWLEDGE SYSTEM
=========================================================

A sophisticated medical AI system that intelligently routes questions to:
- CLINICAL knowledge base (medical procedures, techniques, risks)
- NAVIGATION knowledge base (finding surgeons, costs, appointments)

Architecture: Clean, modular, production-ready with comprehensive error handling.
Author: Advanced AI Systems
Version: 2.0.0 - Production Ready
"""

## ============================
# 📦 CONSOLIDATED IMPORTS
# ============================

import os
import sys
import json
import logging
import pickle
import re
import traceback
import subprocess
import shutil
import tempfile
import concurrent.futures

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum

import numpy as np  # Fixed consistent import alias
import pandas as pd
import torch
import faiss
import nltk
import nltk.data
import uvicorn
import fitz

# ============================
# 📊 DATA STRUCTURES & ENUMS
# ============================

class QueryIntent(Enum):
    """Query intent classification"""
    CLINICAL = "clinical"
    NAVIGATION = "navigation"

class SystemStatus(Enum):
    """System health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

@dataclass
class KnowledgeBase:
    """Represents a knowledge base with its components"""
    chunks: List[str]
    embeddings: Any  # numpy.ndarray
    faiss_index: Any  # faiss.Index

    @property
    def size(self) -> int:
        return len(self.chunks)

@dataclass
class OrganizationData:
    """Complete data structure for an organization"""
    clinical: KnowledgeBase
    navigation: KnowledgeBase

    @property
    def total_chunks(self) -> int:
        return self.clinical.size + self.navigation.size

# ============================
# 🏗️ KNOWLEDGE BASE BUILDER
# ============================

class KnowledgeBaseBuilder:
    """Builds and manages knowledge bases for clinical and navigation content"""
    
    def __init__(self, org_id: str = "asps"):
        self.org_id = org_id
        self.base_path = Path(os.path.dirname(os.path.abspath(__file__))) / "org_data" / org_id
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def build_clinical_knowledge_base(self) -> KnowledgeBase:
        """Build clinical knowledge base from training materials"""
        logger.info("📚 Building clinical knowledge base...")
        
        clinical_chunks = []
        clinical_dirs = [
            Path("clinical"),
            Path("op notes"),
            Path("textbook notes")
        ]
        
        for clinical_dir in clinical_dirs:
            if clinical_dir.exists():
                logger.info(f"✅ Processing clinical directory: {clinical_dir}")
                clinical_chunks.extend(self._process_directory(clinical_dir))
            else:
                logger.info(f"📁 Clinical directory not found: {clinical_dir}")
        
        if not clinical_chunks:
            logger.warning("⚠️ No clinical data found, using placeholder")
            clinical_chunks = [
                "Clinical information will be available soon. "
                "Please consult with a board-certified plastic surgeon for medical guidance."
            ]
        
        return self._build_knowledge_base(clinical_chunks, "clinical")
    
    def build_navigation_knowledge_base(self) -> KnowledgeBase:
        """Build navigation knowledge base from JSON files"""
        logger.info("🧭 Building navigation knowledge base...")
        
        navigation_chunks = []
        json_files = [
            "nav1.json", "nav2.json",
            "comprehensive_split_01.json", "comprehensive_split_02.json",
            "comprehensive_split_03.json", "comprehensive_split_04.json",
            "comprehensive_split_05.json", "comprehensive_split_06.json",
            "comprehensive_split_07.json", "comprehensive_split_08.json",
            "comprehensive_split_09.json", "comprehensive_split_10.json",
            "comprehensive_split_11.json", "comprehensive_split_12.json",
            "comprehensive_split_13.json", "comprehensive_split_14.json",
            "comprehensive_split_15.json"
        ]
        
        for json_file in json_files:
            json_path = Path(json_file)
            if json_path.exists():
                chunks = self._process_json_file(json_path)
                navigation_chunks.extend(chunks)
                logger.info(f"✅ Loaded {len(chunks)} chunks from {json_file}")
            else:
                logger.info(f"📄 JSON file not found: {json_file}")
        
        if not navigation_chunks:
            logger.warning("⚠️ No navigation JSON data found, using fallback")
            navigation_chunks = [
                "To find a qualified plastic surgeon, visit the Find a Surgeon tool on plasticsurgery.org.",
                "For procedure costs and financing options, contact ASPS member surgeons directly.",
                "Before and after photos are available in the photo gallery section.",
                "Patient safety information is available in the Patient Safety section."
            ]
        
        return self._build_knowledge_base(navigation_chunks, "navigation")
    
    def _process_directory(self, directory: Path) -> List[str]:
        """Process all files in a directory"""
        chunks = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                text = self._extract_text_from_file(file_path)
                if text:
                    file_chunks = TextChunker.chunk_text(text)
                    chunks.extend(file_chunks)
                    logger.info(f"📄 Processed {file_path.name}: {len(file_chunks)} chunks")
        
        return chunks
    
    def _extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from various file formats"""
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == ".pdf":
                return DocumentProcessor.extract_text_from_pdf(str(file_path))
            elif suffix in [".docx", ".doc"]:
                return DocumentProcessor.extract_text_from_docx(str(file_path))
            elif suffix in [".png", ".jpg", ".jpeg", ".tiff"]:
                return DocumentProcessor.extract_text_from_image(str(file_path))
            elif suffix in [".html", ".htm"]:
                return DocumentProcessor.extract_text_from_html(str(file_path))
            elif suffix == ".txt":
                return file_path.read_text(encoding="utf-8")
            else:
                logger.warning(f"⚠️ Unsupported file format: {suffix}")
                return ""
        except Exception as e:
            logger.error(f"❌ Failed to process {file_path}: {e}")
            return ""
    
    def _process_json_file(self, json_path: Path) -> List[str]:
        """Process JSON knowledge base file"""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            chunks = []
            
            if isinstance(data, list):
                for item in data:
                    text = self._extract_text_from_json_item(item)
                    if text and len(text.split()) > 20:
                        chunks.append(text)
            elif isinstance(data, dict):
                text = self._extract_text_from_json_item(data)
                if text and len(text.split()) > 20:
                    chunks.append(text)
            
            return chunks
        except Exception as e:
            logger.error(f"❌ Failed to process JSON {json_path}: {e}")
            return []
    
    def _extract_text_from_json_item(self, item: Union[Dict, str]) -> str:
        """Extract text from JSON item"""
        if isinstance(item, str):
            return item.strip()
        elif isinstance(item, dict):
            for field in ['text', 'content', 'description', 'body']:
                if field in item and isinstance(item[field], str):
                    return item[field].strip()
        return ""
    
    def _build_knowledge_base(self, chunks: List[str], kb_type: str) -> KnowledgeBase:
        """Build FAISS knowledge base from chunks"""
        logger.info(f"🔢 Building {kb_type} FAISS index with {len(chunks)} chunks...")
        
        unique_chunks = list(dict.fromkeys(chunks))
        
        embeddings = MODEL_MANAGER.encode_texts(unique_chunks, show_progress_bar=True)
        
        dimension = embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(embeddings.astype(np.float32))  # <-- Use np.float32 consistently
        
        self._save_knowledge_base(unique_chunks, embeddings, faiss_index, kb_type)
        
        logger.info(f"✅ {kb_type.capitalize()} knowledge base built: {len(unique_chunks)} chunks")
        
        return KnowledgeBase(
            chunks=unique_chunks,
            embeddings=embeddings,
            faiss_index=faiss_index
        )
    
    def _save_knowledge_base(self, chunks: List[str], embeddings: Any, faiss_index: Any, kb_type: str) -> None:
        """Save knowledge base components to disk"""
        try:
            chunks_path = self.base_path / f"{kb_type}_chunks.pkl"
            embeddings_path = self.base_path / f"{kb_type}_embeddings.npy"
            index_path = self.base_path / f"{kb_type}_index.faiss"
            
            with open(chunks_path, "wb") as f:
                pickle.dump(chunks, f)
            
            np.save(embeddings_path, embeddings)
            faiss.write_index(faiss_index, str(index_path))
            
            logger.info(f"💾 {kb_type.capitalize()} knowledge base saved to disk")
        except Exception as e:
            logger.error(f"❌ Failed to save {kb_type} knowledge base: {e}")

# ============================
# 🔍 INTELLIGENT RETRIEVAL SYSTEM
# ============================

class RetrievalSystem:
    """Advanced retrieval system with quality scoring"""
    
    @staticmethod
    def retrieve_context(
        query: str, 
        intent: QueryIntent, 
        org_id: str = "asps", 
        k: int = None
    ) -> List[str]:
        """Retrieve relevant context based on query intent"""
        k = k or CONFIG.RETRIEVAL_K
        
        if not query.strip():
            return []
        
        if org_id not in ORGANIZATIONS:
            logger.error(f"❌ Organization '{org_id}' not found")
            return []
        
        org_data = ORGANIZATIONS[org_id]
        knowledge_base = (
            org_data.clinical if intent == QueryIntent.CLINICAL 
            else org_data.navigation
        )
        
        if knowledge_base.size == 0:
            logger.warning(f"⚠️ Empty {intent.value} knowledge base")
            return []
        
        try:
            # Generate query embedding
            query_embedding = MODEL_MANAGER.encode_texts([query])[0].reshape(1, -1)
            
            # Search FAISS index
            distances, indices = knowledge_base.faiss_index.search(
                query_embedding.astype(numpy.float32), 
                min(CONFIG.INITIAL_K, knowledge_base.size)
            )
            
            # Calculate semantic similarity scores
            candidates = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(knowledge_base.chunks):
                    chunk = knowledge_base.chunks[idx]
                    chunk_embedding = knowledge_base.embeddings[idx].reshape(1, -1)
                    similarity = cosine_similarity(query_embedding, chunk_embedding)[0][0]
                    
                    # Apply quality threshold
                    if similarity > CONFIG.SIMILARITY_THRESHOLD:
                        candidates.append((chunk, similarity))
            
            # Sort by similarity and return top k
            candidates.sort(key=lambda x: x[1], reverse=True)
            results = [chunk for chunk, _ in candidates[:k]]
            
            logger.info(f"🔍 Retrieved {len(results)} {intent.value} chunks")
            return results
            
        except Exception as e:
            logger.error(f"❌ Retrieval failed for {intent.value}: {e}")
            return []

# ============================
# 🎨 RESPONSE GENERATION SYSTEM
# ============================

class ResponseGenerator:
    """Intelligent response generation with safety mechanisms"""
    
    SAFETY_MESSAGES = {
        QueryIntent.CLINICAL: (
            "I prioritize your safety and health by not providing medical information "
            "I cannot verify from my training materials. Rather than risk giving you "
            "incorrect clinical guidance, I strongly recommend:\n\n"
            "🩺 **Consult a board-certified plastic surgeon** - They can provide "
            "personalized, evidence-based advice\n"
            "📚 **Review peer-reviewed medical literature** - Look for recent studies "
            "on your specific concern\n"
            "🏥 **Speak with your healthcare provider** - They know your medical "
            "history and current health status\n"
            "📞 **Contact ASPS** at (847) 228-9900 for surgeon referrals in your area\n\n"
            "Your health and safety are paramount - professional medical consultation "
            "is always the safest approach for clinical questions."
        ),
        QueryIntent.NAVIGATION: (
            "I don't have specific information about this ASPS website navigation "
            "question in my current knowledge base. Rather than provide potentially "
            "outdated guidance, I recommend:\n\n"
            "📍 Visit plasticsurgery.org directly for current information\n"
            "🔍 Use their site search function for specific topics\n"
            "📞 Contact ASPS support at (847) 228-9900 for personalized assistance\n\n"
            "This ensures you get the most accurate and up-to-date information about "
            "their website features and resources."
        )
    }
    
    @classmethod
    def generate_response(
        cls, 
        query: str, 
        context_chunks: List[str], 
        intent: QueryIntent
    ) -> str:
        """Generate intelligent response based on context and intent"""
        
        # Return safety message if no context
        if not context_chunks:
            return cls.SAFETY_MESSAGES[intent]
        
        # Build context
        context = "\n\n".join(f"- {chunk.strip()}" for chunk in context_chunks)
        
        # Create intent-specific prompt
        prompt = cls._build_prompt(query, context, intent)
        
        # Generate response
        try:
            response = MODEL_MANAGER.generate_response(prompt)
            
            # Validate response quality
            if cls._validate_response_quality(response, context):
                return response
            else:
                logger.warning("⚠️ Response failed quality validation")
                return cls.SAFETY_MESSAGES[intent]
                
        except Exception as e:
            logger.error(f"❌ Response generation failed: {e}")
            return cls.SAFETY_MESSAGES[intent]
    
    @staticmethod
    def _build_prompt(query: str, context: str, intent: QueryIntent) -> str:
        """Build intent-specific prompt"""
        
        if intent == QueryIntent.CLINICAL:
            return (
                "You are a knowledgeable medical professional providing educational "
                "information about plastic surgery. Write your response in a clear, "
                "professional yet conversational tone. Use only the CONTEXT below to "
                "provide accurate medical information. Explain things in a way that's "
                "informative but accessible to patients. Focus on being helpful and "
                "educational while maintaining professional medical standards.\n\n"
                f"### CONTEXT:\n{context}\n\n"
                f"### QUESTION:\n{query}\n\n"
                f"### ANSWER:\n"
            )
        else:
            return (
                "You are a knowledgeable and helpful assistant providing guidance "
                "about ASPS resources and services. Write your response in a natural, "
                "conversational tone. Use only the CONTEXT below to provide clear, "
                "actionable information. Be specific and helpful, giving users "
                "practical steps they can take. Keep your response focused, "
                "informative, and easy to follow.\n\n"
                f"### CONTEXT:\n{context}\n\n"
                f"### QUESTION:\n{query}\n\n"
                f"### ANSWER:\n"
            )
    
    @staticmethod
    def _validate_response_quality(response: str, context: str) -> bool:
        """Validate response quality using token overlap"""
        response_tokens = set(re.findall(r"\b\w+\b", response.lower()))
        context_tokens = set(re.findall(r"\b\w+\b", context.lower()))
        
        if not response_tokens:
            return False
        
        overlap = response_tokens & context_tokens
        overlap_score = len(overlap) / len(response_tokens)
        
        return overlap_score >= CONFIG.HALLUCINATION_THRESHOLD

# ============================
# 🏥 MAIN ASPS SYSTEM
# ============================

class ASPSMedicalSystem:
    """Main ASPS Medical AI System"""
    
    def __init__(self, org_id: str = "asps"):
        self.org_id = org_id
        self.initialized = False
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the complete ASPS system"""
        try:
            logger.info("🚀 Initializing ASPS Medical AI System...")
            
            # Step 1: GitHub Data Sync (if enabled)
            if GITHUB_LOADER.enabled:
                logger.info("🐙 GitHub integration enabled - syncing training data...")
                sync_stats = GITHUB_LOADER.sync_training_data()
                
                if sync_stats["clinical_files"] > 0 or sync_stats["navigation_files"] > 0:
                    logger.info(f"✅ GitHub sync successful: {sync_stats}")
                else:
                    logger.warning("⚠️ GitHub sync completed but no files were synced")
            else:
                logger.info("📁 Using local training data (GitHub token not provided)")
            
            # Step 2: Build knowledge bases
            builder = KnowledgeBaseBuilder(self.org_id)
            clinical_kb = builder.build_clinical_knowledge_base()
            navigation_kb = builder.build_navigation_knowledge_base()
            
            # Store in global registry
            ORGANIZATIONS[self.org_id] = OrganizationData(
                clinical=clinical_kb,
                navigation=navigation_kb
            )
            
            self.initialized = True
            
            # Log system status
            total_chunks = ORGANIZATIONS[self.org_id].total_chunks
            logger.info(f"✅ ASPS System initialized successfully!")
            logger.info(f"📊 Clinical chunks: {clinical_kb.size}")
            logger.info(f"📊 Navigation chunks: {navigation_kb.size}")
            logger.info(f"📊 Total knowledge base: {total_chunks} chunks")
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            raise RuntimeError("Critical system initialization failure")
    
    def query(self, question: str) -> Dict[str, Any]:
        """Process a user query and return structured response"""
        if not self.initialized:
            raise RuntimeError("System not initialized")
        
        try:
            # Classify intent
            intent = IntentClassifier.classify(question)
            
            # Retrieve context
            context_chunks = RetrievalSystem.retrieve_context(
                query=question,
                intent=intent,
                org_id=self.org_id
            )
            
            # Generate response
            answer = ResponseGenerator.generate_response(
                query=question,
                context_chunks=context_chunks,
                intent=intent
            )
            
            return {
                "question": question,
                "intent": intent.value,
                "answer": answer,
                "context_chunks": context_chunks,
                "system_status": "healthy"
            }
            
        except Exception as e:
            logger.error(f"❌ Query processing failed: {e}")
            return {
                "question": question,
                "intent": "unknown",
                "answer": "I apologize, but I'm experiencing technical difficulties. Please try again later.",
                "context_chunks": [],
                "system_status": "error",
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        status = {
            "system": SystemStatus.HEALTHY.value,
            "components": {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Check models
        try:
            if MODEL_MANAGER.embedding_model is None:
                status["components"]["embedding_model"] = {"status": "error", "message": "Not loaded"}
                status["system"] = SystemStatus.UNHEALTHY.value
            else:
                # Test embedding
                MODEL_MANAGER.encode_texts(["test"])
                status["components"]["embedding_model"] = {
                    "status": "healthy", 
                    "device": str(MODEL_MANAGER.embedding_model.device)
                }
        except Exception as e:
            status["components"]["embedding_model"] = {"status": "error", "message": str(e)}
            status["system"] = SystemStatus.UNHEALTHY.value
        
        # Check language model
        try:
            if MODEL_MANAGER.language_model is None:
                status["components"]["language_model"] = {"status": "error", "message": "Not loaded"}
                status["system"] = SystemStatus.UNHEALTHY.value
            else:
                status["components"]["language_model"] = {
                    "status": "healthy",
                    "device": str(MODEL_MANAGER.language_model.device)
                }
        except Exception as e:
            status["components"]["language_model"] = {"status": "error", "message": str(e)}
            status["system"] = SystemStatus.UNHEALTHY.value
        
        # Check knowledge bases
        if self.org_id in ORGANIZATIONS:
            org_data = ORGANIZATIONS[self.org_id]
            status["components"]["knowledge_bases"] = {
                "status": "healthy",
                "clinical_chunks": org_data.clinical.size,
                "navigation_chunks": org_data.navigation.size,
                "total_chunks": org_data.total_chunks
            }
        else:
            status["components"]["knowledge_bases"] = {"status": "error", "message": "Not loaded"}
            status["system"] = SystemStatus.UNHEALTHY.value
        
        # Check GitHub integration
        github_status = GITHUB_LOADER.check_github_status()
        status["components"]["github"] = github_status
        
        # Check CUDA
        if torch.cuda.is_available():
            try:
                memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                status["components"]["cuda"] = {
                    "status": "available",
                    "device_count": torch.cuda.device_count(),
                    "memory_allocated_gb": round(memory_allocated, 2),
                    "memory_reserved_gb": round(memory_reserved, 2)
                }
            except Exception as e:
                status["components"]["cuda"] = {"status": "error", "message": str(e)}
        else:
            status["components"]["cuda"] = {"status": "not_available"}
        
        return status

# ============================
# 🌐 FASTAPI WEB INTERFACE
# ============================

# Initialize ASPS system
asps_system = ASPSMedicalSystem()

# Create FastAPI app
app = FastAPI(
    title="🏥 ASPS Medical AI Chatbot",
    description="Intelligent medical AI system with clinical/navigation dual knowledge routing",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: Optional[int] = CONFIG.RETRIEVAL_K

class QueryResponse(BaseModel):
    question: str
    intent: str
    answer: str
    context_chunks: List[str]
    system_status: str

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def chatbot_ui():
    """Serve the chatbot web interface"""
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏥 ASPS Medical AI Chatbot</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh; padding: 20px;
            }
            .container { 
                max-width: 800px; margin: 0 auto; 
                background: white; border-radius: 20px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1); overflow: hidden;
            }
            .header { 
                background: linear-gradient(135deg, #2196F3, #21CBF3);
                color: white; padding: 30px; text-align: center;
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { font-size: 1.1em; opacity: 0.9; }
            .chat-container { padding: 30px; max-height: 500px; overflow-y: auto; }
            .input-container { padding: 20px 30px 30px; }
            .input-group { display: flex; gap: 10px; }
            input[type="text"] { 
                flex: 1; padding: 15px; border: 2px solid #e0e0e0; 
                border-radius: 25px; font-size: 16px; outline: none;
                transition: border-color 0.3s;
            }
            input[type="text"]:focus { border-color: #2196F3; }
            button { 
                padding: 15px 30px; background: linear-gradient(135deg, #2196F3, #21CBF3);
                color: white; border: none; border-radius: 25px; 
                cursor: pointer; font-size: 16px; transition: transform 0.2s;
            }
            button:hover { transform: translateY(-2px); }
            .message { margin: 15px 0; padding: 15px; border-radius: 15px; }
            .user-message { background: #e3f2fd; margin-left: 50px; }
            .bot-message { background: #f5f5f5; margin-right: 50px; }
            .intent-badge { 
                display: inline-block; padding: 5px 10px; border-radius: 10px;
                font-size: 12px; font-weight: bold; margin-bottom: 10px;
            }
            .clinical { background: #ffebee; color: #c62828; }
            .navigation { background: #e8f5e8; color: #2e7d32; }
            .loading { text-align: center; padding: 20px; color: #666; }
            .footer { text-align: center; padding: 20px; color: #666; font-size: 14px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏥 ASPS Medical AI</h1>
                <p>Intelligent Clinical & Navigation Assistant</p>
            </div>
            
            <div class="chat-container" id="chatContainer">
                <div class="message bot-message">
                    <strong>🤖 ASPS AI Assistant</strong><br>
                    Hello! I'm your intelligent medical AI assistant. I can help with:
                    <br><br>
                    🩺 <strong>Clinical Questions:</strong> Medical procedures, techniques, risks, recovery
                    <br>
                    🧭 <strong>Navigation Questions:</strong> Finding surgeons, costs, appointments, website help
                    <br><br>
                    What would you like to know?
                </div>
            </div>
            
            <div class="input-container">
                <div class="input-group">
                    <input type="text" id="questionInput" placeholder="Ask me about plastic surgery procedures or ASPS services..." onkeypress="handleEnter(event)">
                    <button onclick="askQuestion()">Ask</button>
                </div>
            </div>
            
            <div class="footer">
                <p>🔒 For medical advice, always consult with a board-certified plastic surgeon</p>
            </div>
        </div>

        <script>
            async function askQuestion() {
                const input = document.getElementById('questionInput');
                const question = input.value.trim();
                if (!question) return;

                const chatContainer = document.getElementById('chatContainer');
                
                // Add user message
                chatContainer.innerHTML += `
                    <div class="message user-message">
                        <strong>👤 You:</strong><br>${question}
                    </div>
                `;
                
                // Add loading message
                chatContainer.innerHTML += `
                    <div class="message bot-message loading" id="loading">
                        <strong>🤖 ASPS AI:</strong><br>🔍 Analyzing your question...
                    </div>
                `;
                
                input.value = '';
                chatContainer.scrollTop = chatContainer.scrollHeight;

                try {
                    const response = await fetch('/query', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question: question })
                    });

                    const data = await response.json();
                    
                    // Remove loading message
                    document.getElementById('loading').remove();
                    
                    // Add bot response
                    const intentClass = data.intent === 'clinical' ? 'clinical' : 'navigation';
                    const intentIcon = data.intent === 'clinical' ? '🩺' : '🧭';
                    
                    chatContainer.innerHTML += `
                        <div class="message bot-message">
                            <div class="intent-badge ${intentClass}">
                                ${intentIcon} ${data.intent.toUpperCase()} QUERY
                            </div>
                            <strong>🤖 ASPS AI:</strong><br>${data.answer.replace(/\\n/g, '<br>')}
                        </div>
                    `;
                    
                } catch (error) {
                    document.getElementById('loading').remove();
                    chatContainer.innerHTML += `
                        <div class="message bot-message">
                            <strong>🤖 ASPS AI:</strong><br>
                            ❌ I'm experiencing technical difficulties. Please try again later.
                        </div>
                    `;
                }
                
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function handleEnter(event) {
                if (event.key === 'Enter') {
                    askQuestion();
                }
            }
        </script>
    </body>
    </html>
    """

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint"""
    try:
        result = asps_system.query(request.question)
        return QueryResponse(**result)
    except Exception as e:
        logger.error(f"❌ Query endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health")
async def health_check():
    """System health check endpoint"""
    return asps_system.get_health_status()

@app.get("/github-sync")
async def github_sync_endpoint():
    """Manually trigger GitHub data sync"""
    if not GITHUB_LOADER.enabled:
        raise HTTPException(
            status_code=400, 
            detail="GitHub integration not configured. Please set GITHUB_TOKEN environment variable."
        )
    
    try:
        logger.info("🔄 Manual GitHub sync triggered via API")
        sync_stats = GITHUB_LOADER.sync_training_data()
        
        return {
            "status": "success",
            "message": "GitHub sync completed",
            "stats": sync_stats,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Manual GitHub sync failed: {e}")
        raise HTTPException(status_code=500, detail=f"GitHub sync failed: {str(e)}")

@app.get("/github-status")
async def github_status_endpoint():
    """Get GitHub integration status"""
    return GITHUB_LOADER.check_github_status()

@app.get("/sample-questions")
async def get_sample_questions():
    """Get sample questions for testing"""
    return {
        "clinical_questions": [
            "What are the typical indications for tissue expander placement in breast reconstruction?",
            "What is the vascular supply of the radial forearm flap?",
            "Explain the operative steps for a free TRAM flap breast reconstruction.",
            "What precautions are needed to avoid peroneal nerve injury during fibula flap harvest?",
            "How is capsulorrhaphy performed during implant exchange in breast reconstruction?"
        ],
        "navigation_questions": [
            "How can I find a board-certified plastic surgeon near me?",
            "What should I expect in terms of costs for breast augmentation?",
            "Where can I see before and after photos of rhinoplasty procedures?",
            "How do I schedule a consultation with an ASPS member surgeon?",
            "What questions should I ask during my plastic surgery consultation?"
        ]
    }

# ============================
# 🚀 MAIN EXECUTION
# ============================

def main():
    """Main execution function"""
    
    print("🏥 ASPS MEDICAL AI CHATBOT - ROBUST DUAL KNOWLEDGE SYSTEM")
    print("=" * 65)
    print("🎯 System Architecture:")
    print("   • Intelligent Intent Classification (Clinical vs Navigation)")
    print("   • Dual Knowledge Base System with FAISS Indexing")
    print("   • Advanced Retrieval with Semantic Similarity")
    print("   • Mistral-7B Powered Response Generation")
    print("   • Comprehensive Error Handling & Recovery")
    print("   • Production-Ready FastAPI Interface")
    print("   • GitHub Integration for Automatic Data Sync")
    print("")
    
    # GitHub status check
    github_status = GITHUB_LOADER.check_github_status()
    if github_status["enabled"]:
        print("🐙 GitHub Integration: ENABLED")
        print(f"   Repository: {github_status['repository_url']}")
        print(f"   Branch: {github_status['branch']}")
        print(f"   Git Available: {github_status['git_available']}")
    else:
        print("📁 GitHub Integration: DISABLED (using local data)")
        print("   💡 Set GITHUB_TOKEN environment variable to enable")
    print("")
    
    # System health check
    health = asps_system.get_health_status()
    if health["system"] == SystemStatus.HEALTHY.value:
        print("✅ System Status: HEALTHY")
        print(f"📊 Knowledge Base: {health['components']['knowledge_bases']['total_chunks']} total chunks")
        print(f"   - Clinical: {health['components']['knowledge_bases']['clinical_chunks']} chunks")
        print(f"   - Navigation: {health['components']['knowledge_bases']['navigation_chunks']} chunks")
        print("")
        print("🌐 Starting web server...")
        print("📍 Server available at: http://localhost:8000")
        print("📖 API documentation: http://localhost:8000/docs")
        print("🔍 Health check: http://localhost:8000/health")
        if github_status["enabled"]:
            print("🐙 GitHub sync: http://localhost:8000/github-sync")
            print("📊 GitHub status: http://localhost:8000/github-status")
        print("")
        print("✅ SYSTEM READY FOR INTELLIGENT MEDICAL ASSISTANCE!")
        
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ System Status: UNHEALTHY")
        print("🚨 Please check system logs and resolve issues before starting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
