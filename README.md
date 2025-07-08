# Athena Surgical RAG API

**Version:** 2.0  
**Author:** Sophie A. Wolmer  
**Last Updated:** July 2025  
**Domain Focus:** Multi-Organization, Physician-Empowered AI for Advanced Surgical Intelligence & Clinical Decision Support  
*(Built for Plastic and Reconstructive Surgery — extensible to all medical and surgical specialties)*  
**Language Model:** Hermes-2-Pro (Mistral-7B) with Fine-tuning Capabilities

---

## Executive Summary

Athena Surgical RAG API is a next-generation, multi-organizational RAG platform that enables any institution to deploy isolated, custom AI chatbots while maintaining complete data sovereignty. This flexible system allows organizations to create personalized AI assistants using their own training materials, custom branding, and domain-specific knowledge—eliminating vendor lock-in and ensuring ethical, transparent, and organization-specific intelligence.

Key innovations include:
- **Multi-tenant architecture** supporting unlimited custom organizations with any naming convention
- **Dynamic ZIP upload and indexing** for real-time training material integration  
- **Custom fine-tuning pipeline** with Hugging Face Trainer integration
- **Hallucination detection** via token-level overlap scoring
- **CUDA-optimized processing** for enterprise-grade performance
- **Comprehensive evaluation framework** with structured output validation

Leveraging a state-of-the-art 7B parameter large language model with organization-specific fine-tuning, Athena delivers context-aware, evidence-based responses while maintaining complete institutional data isolation.

---

## Primary Use Cases

- **Custom organizational chatbots** with domain-specific knowledge bases
- **Educational institutions** with curriculum-specific AI assistants
- **Corporate training platforms** with company-specific content integration
- **Research organizations** with specialized literature and protocol access
- **Healthcare institutions** with medical protocol and guideline integration
- **Legal firms** with case law and regulatory document access
- **Consulting companies** with proprietary methodology and best practice databases
- **Any organization** requiring isolated, custom AI assistants with their own training materials

---

## Architecture Overview

### Core Infrastructure
- **Language Model:** `NousResearch/Hermes-2-Pro-Mistral-7B` with custom fine-tuning capabilities
- **Embedding Model:** `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net/)
- **Vector Indexing:** [FAISS](https://github.com/facebookresearch/faiss) (Flat L2 index, per-organization isolation)
- **Training Framework:** Hugging Face Transformers with custom Trainer class
- **API Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.10+)

### Multi-Organization Support
- **Data Isolation:** Complete separation of training materials, indexes, and models per organization
- **Dynamic Scaling:** Unlimited organization support with automatic resource management
- **Storage Architecture:** Hierarchical file system with org-specific directories (`org_data/{org_id}/`)

### Document Processing Pipeline
- **Supported Formats:** `.pdf`, `.docx`, `.jpg`/`.png` (via Tesseract OCR)
- **Text Extraction:** PyMuPDF, python-docx, PIL + pytesseract
- **Chunking Strategy:** Word-based overlapping chunks (200 words, 50-word overlap)
- **Quality Filtering:** Advanced content validation and noise removal

### Hallucination Detection
- **Token Overlap Scoring:** Real-time analysis of answer-context alignment
- **Threshold-based Filtering:** Automatic response rejection for low-confidence outputs
- **Evaluation Framework:** Comprehensive assessment with structured output validation

### Hardware Optimization
- **CUDA Acceleration:** Full GPU support for training and inference
- **Memory Management:** Efficient tensor handling with FP16 precision
- **Batch Processing:** Optimized document ingestion and embedding generation

---

## API Endpoints

### `POST /query`

Processes clinical/surgical questions using organization-specific RAG with hallucination detection.

**Headers:**
- `x-api-key`: Required if `DEBUG=False`
- `x-org-id`: Custom organization identifier (e.g., `my-company`, `research-lab-001`, `university-cs-dept`)

**Request Body:**
```json
{
  "question": "How is a DIEP flap performed?",
  "org_id": "my-medical-practice",
  "retrieval_k": 3
}
```

**Response:**
```json
{
  "question": "How is a DIEP flap performed?",
  "answer": "✅ Summary: DIEP flap is a microsurgical breast reconstruction...",
  "metadata": {
    "org_id": "my-medical-practice",
    "overlap_score": 0.76,
    "hallucinated": false,
    "response_time_sec": 1.42,
    "context_chunks_used": 3
  }
}
```

### `POST /upload_materials`

Uploads ZIP archives of training materials for organization-specific indexing.

**Headers:**
- `x-api-key`: Required
- `x-org-id`: Custom organization identifier

**Request:**
- **File Upload:** ZIP archive containing PDF, DOCX, or image files
- **Auto-Processing:** Automatic extraction, chunking, embedding, and FAISS indexing

**Response:**
```json
{
  "status": "success",
  "org_id": "tech-startup-docs",
  "documents_processed": 47,
  "chunks_indexed": 2834,
  "processing_time_sec": 23.7
}
```

### `POST /fine_tune`

Initiates custom model fine-tuning using organization-specific datasets.

**Headers:**
- `x-api-key`: Required
- `x-org-id`: Custom organization identifier

**Request Body:**
```json
{
  "jsonl_path": "training_data/company_qa.jsonl",
  "epochs": 3,
  "learning_rate": 2e-4,
  "batch_size": 1
}
```

### `GET /analytics/dashboard`

Returns comprehensive usage metrics and model performance analytics.

**Query Parameters:**
- `org_id` (optional): Filter by organization
- `timeframe` (optional): `7d`, `30d`, `all`

**Response:**
```json
{
  "total_queries": 1247,
  "hallucination_rate": 0.063,
  "avg_response_time_sec": 1.72,
  "avg_overlap_score": 0.81,
  "top_organizations": ["acme-corp", "university-research", "consulting-firm"],
  "model_versions": {
    "base": 892,
    "fine_tuned": 355
  }
}
```

### `GET /organizations/{org_id}/status`

Returns organization-specific system status and resource information.

**Response:**
```json
{
  "org_id": "my-startup",
  "index_status": "ready",
  "document_count": 1423,
  "chunk_count": 45671,
  "last_updated": "2025-07-08T14:23:11Z",
  "model_status": "fine_tuned",
  "available_endpoints": ["query", "upload", "fine_tune"]
}
```

---

## Environment Configuration

Create a `.env` file at the project root with the following configuration:

```env
# Required API Configuration
RAG_API_KEY=your_secure_api_key_here

# Optional Hugging Face Token (for private models)
HF_TOKEN=your_huggingface_token_here

# System Configuration
DEBUG=False
CUDA_VISIBLE_DEVICES=0
MAX_ORGANIZATIONS=100

# Model Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
LLM_MODEL_NAME=NousResearch/Hermes-2-Pro-Mistral-7B

# Storage Configuration
ORG_DATA_ROOT=./org_data
MAX_UPLOAD_SIZE_MB=500
```

---

## Setup Instructions

### 1. System Requirements
- **Python:** 3.10 or higher
- **GPU:** CUDA-compatible GPU with 8GB+ VRAM (recommended)
- **RAM:** 16GB+ system memory
- **Storage:** 50GB+ available space for models and indices

### 2. Installation
```bash
# Clone repository
git clone https://github.com/your-org/athena-surgical-rag.git
cd athena-surgical-rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.\.venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for OCR
# Ubuntu/Debian:
sudo apt-get install tesseract-ocr
# macOS:
brew install tesseract
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

### 3. Initialize Organization Data
```bash
# Create custom organization directories (use any naming convention)
python main_script.py --create_org my-company
python main_script.py --create_org research-lab-alpha
python main_script.py --create_org university-cs-dept

# Upload training materials (example)
python main_script.py --upload_zip company_policies.zip --org_id my-company

# Build initial FAISS index
python main_script.py --build_index --org_id my-company
```

### 4. Start API Server
```bash
# Development mode
uvicorn api_server:app --reload --host 0.0.0.0 --port 8000

# Production mode
uvicorn api_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Run Evaluation (Optional)
```bash
# Evaluate model performance on test questions
python main_script.py --evaluate --org_id my-company --output eval_results.json
```

---

## Testing Examples

### Basic Query (via curl)
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -H "x-org-id: my-company" \
  -d '{"question": "What is our company policy on remote work?", "retrieval_k": 3}'
```

### Upload Training Materials
```bash
curl -X POST http://127.0.0.1:8000/upload_materials \
  -H "x-api-key: your_secure_api_key_here" \
  -H "x-org-id: my-company" \
  -F "file=@company_documents.zip"
```

### Check Organization Status
```bash
curl -X GET http://127.0.0.1:8000/organizations/my-company/status \
  -H "x-api-key: your_secure_api_key_here"
```

### Fine-tune Model
```bash
curl -X POST http://127.0.0.1:8000/fine_tune \
  -H "Content-Type: application/json" \
  -H "x-api-key: your_secure_api_key_here" \
  -H "x-org-id: my-company" \
  -d '{"jsonl_path": "company_qa.jsonl", "epochs": 3}'
```

### Analytics Dashboard
```bash
curl -X GET "http://127.0.0.1:8000/analytics/dashboard?org_id=my-company&timeframe=30d" \
  -H "x-api-key: your_secure_api_key_here"
```

---

## Advanced Features

### Multi-Organization Architecture
- **Complete Data Isolation:** Each organization maintains separate indexes, embeddings, and models
- **Dynamic Resource Management:** Automatic scaling and memory optimization per organization
- **Collaborative Research Mode:** Future support for cross-organizational knowledge sharing with permission controls

### Intelligent Document Processing
- **Advanced Text Extraction:** Multi-format support with OCR capabilities for images and scanned documents
- **Smart Chunking:** Context-aware text segmentation with configurable overlap strategies
- **Quality Filtering:** Automatic removal of headers, footers, copyright notices, and low-quality content

### Hallucination Detection & Validation
- **Token-Level Overlap Analysis:** Real-time comparison of generated answers with source context
- **Confidence Thresholding:** Automatic response filtering based on evidence alignment scores
- **Structured Output Validation:** Ensures clinical answers follow medical formatting standards

### Model Fine-Tuning Pipeline
- **Custom Training:** Organization-specific model adaptation using institutional datasets
- **Hyperparameter Optimization:** Automated tuning for optimal performance per medical specialty
- **Evaluation Framework:** Comprehensive assessment with medical accuracy metrics

### Enterprise Integration
- **RESTful API Design:** Seamless integration with existing clinical systems and EMRs
- **Scalable Architecture:** Horizontal scaling support for high-volume clinical environments
- **Security Framework:** Role-based access control and audit logging for healthcare compliance

## Performance Metrics

### Benchmark Results (Example Institution)
- **Query Response Time:** 1.2-2.1 seconds average
- **Hallucination Rate:** <6.5% with overlap scoring enabled
- **Index Build Time:** ~45 seconds per 1,000 document pages
- **Memory Usage:** 8-12GB VRAM for 7B parameter model
- **Throughput:** 50-100 concurrent queries supported

---

## Directory Structure

```
athena-surgical-rag/
├── main_script.py              # Core RAG implementation with multi-org support
├── api_server.py               # FastAPI server with organization isolation
├── requirements.txt            # Python dependencies
├── .env                        # Environment configuration
├── README.md                   # This documentation
│
├── org_data/                   # Organization-specific data storage
│   ├── my-company/             # Example: Custom company chatbot
│   │   ├── training/           # Raw training materials (PDFs, DOCX, images)
│   │   ├── model/              # Fine-tuned model artifacts
│   │   ├── faiss_index.idx     # FAISS vector index
│   │   ├── rag_chunks.pkl      # Processed text chunks
│   │   ├── rag_embeddings.npy  # Document embeddings
│   │   └── Training_QA_Pairs.csv # Optional structured training data
│   │
│   ├── research-lab/           # Example: Research institution
│   │   ├── training/
│   │   ├── model/
│   │   └── ...
│   │
│   └── university-dept/        # Example: University department
│       ├── training/
│       ├── model/
│       └── ...
│
├── shared_models/              # Global model cache
│   ├── hermes-2-pro/          # Base language model
│   └── sentence-transformers/ # Embedding models
│
├── logs/                       # Application logs
├── eval_outputs/              # Evaluation results
└── temp/                      # Temporary file processing
```

## Compliance & Security

### Data Protection & Privacy
- **Complete Data Sovereignty:** Each organization maintains full control over their data and models
- **Flexible Compliance:** Adaptable to various industry requirements (HIPAA, GDPR, SOX, etc.)
- **Data Encryption:** At-rest and in-transit encryption for all organizational data
- **Access Control:** Role-based permissions with audit trail logging
- **Data Retention:** Configurable retention policies per organizational requirements

### AI Ethics & Transparency
- **Explainable AI:** Token-level attribution and source document traceability
- **Bias Detection:** Monitoring for demographic and domain-specific biases
- **Human Oversight:** Configurable validation requirements for high-stakes decisions
- **Version Control:** Complete model versioning and rollback capabilities

### Regulatory Considerations
- **Industry Agnostic:** Designed to support various regulatory frameworks
- **Audit Trail:** Comprehensive logging for compliance reporting
- **Customizable Controls:** Configurable security and validation measures per organization
- **Documentation Framework:** Built-in support for compliance documentation and validation studies

## License and Use

This software and all associated materials are proprietary to Athena AI Platform and developed by Sophie A. Wolmer. The codebase is intended for creating custom AI chatbots for organizations across various industries and use cases.

### Usage Restrictions
- Unauthorized copying, redistribution, modification, or deployment is strictly prohibited
- Commercial use requires explicit written permission from the Athena AI Platform development team
- Users must comply with all applicable regulations for their specific industry when deploying in production
- All organizational data remains under the exclusive control and ownership of the respective institution

### Compliance Responsibility
- Users are responsible for ensuring compliance with their industry's regulations (HIPAA, GDPR, SOX, etc.)
- Production deployment requires appropriate organizational oversight and validation procedures
- Regular security audits and data protection assessments are strongly recommended
- Organizations must implement appropriate access controls and monitoring for their specific use case

### Contact Information
For licensing inquiries, commercial partnerships, technical support, or compliance questions:

**Athena AI Platform Development Team**  
**Lead Developer:** Sophie A. Wolmer  
**Email:** swolmer@emory.edu  
**Website:** www.linkedin.com/in/sophie-wolmer-a44bb5a4

---

*Built with ❤️ for organizations worldwide seeking custom AI solutions*

