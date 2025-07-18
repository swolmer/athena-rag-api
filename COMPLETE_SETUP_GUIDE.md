# Complete Athen.ai Medical AI System - Setup & Integration Guide

**From Zero to Production-Ready Medical AI in 30 Minutes**

---

## Table of Contents

1. [Overview](#overview)
2. [Part 1: RunPod Deployment](#part-1-runpod-deployment)
3. [Part 2: GitHub Upload](#part-2-github-upload)
4. [Part 3: Frontend Integration](#part-3-frontend-integration)
5. [Part 4: Testing & Validation](#part-4-testing--validation)
6. [Part 5: Advanced Features](#part-5-advanced-features)
7. [Troubleshooting](#troubleshooting)
8. [Appendix](#appendix)

---

## Overview

This guide walks you through setting up a complete medical AI system with:
- **GPU-powered backend** on RunPod (RTX 4090)
- **Advanced medical RAG** with document processing
- **Fine-tuning capabilities** for organization-specific training
- **Multi-tenant architecture** for multiple hospitals
- **Complete frontend integration** with any existing website

**Total Setup Time: ~30 minutes**
**Technical Level: Intermediate**
**Prerequisites: Basic HTML/JavaScript knowledge**

---

## Part 1: RunPod Deployment

### Step 1.1: Create RunPod Account
1. Go to https://runpod.io
2. Sign up for an account
3. Add payment method (you'll need credits for GPU usage)
4. Navigate to "Pods" section

### Step 1.2: Deploy Your Pod
1. Click **"+ Deploy"**
2. **Template Selection**:
   - Choose **"PyTorch 2.1"** template
   - Or search for "pytorch" and select latest version

3. **GPU Configuration**:
   - **Recommended**: RTX 4090 (24GB VRAM) - Best performance
   - **Alternative**: RTX 3090 (24GB VRAM) - Good performance
   - **Budget**: RTX 3080 (12GB VRAM) - Minimum requirements

4. **Container Settings**:
   - **Container Disk**: 50GB minimum (recommended: 100GB)
   - **Volume Disk**: 20GB (for persistent storage)
   - **Ports**: Expose port 8000 (HTTP)

5. **Environment Variables** (Optional):
   ```
   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
   ```

6. Click **"Deploy On Demand"**

### Step 1.3: Access Your Pod
1. Wait for pod to reach "Running" status (2-3 minutes)
2. Click **"Connect"** → **"Web Terminal"**
3. You'll see a terminal interface

### Step 1.4: Install the Medical AI System
Copy and paste these commands one by one:

```bash
# Update system
apt update && apt install -y wget unzip

# Download the complete system
wget https://github.com/YOUR-USERNAME/YOUR-REPO/archive/main.zip
unzip main.zip
cd YOUR-REPO-main

# Install Python dependencies
pip install -r requirements.txt

# Download and setup models (this takes 5-10 minutes)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import nltk

print('Downloading language model...')
tokenizer = AutoTokenizer.from_pretrained('NousResearch/Hermes-2-Pro-Mistral-7B')
model = AutoModelForCausalLM.from_pretrained('NousResearch/Hermes-2-Pro-Mistral-7B')

print('Downloading embedding model...')
embedder = SentenceTransformer('all-MiniLM-L6-v2')

print('Downloading NLTK data...')
nltk.download('punkt')
nltk.download('stopwords')

print('Setup complete!')
"

# Start the API server
python main_script_orgid.py
```

### Step 1.5: Verify Deployment
1. Look for this message in terminal:
   ```
   INFO: Uvicorn running on http://0.0.0.0:8000
   GPU Memory: XX.X GB / 24.0 GB
   Models loaded successfully!
   ```

2. Note your **Pod URL** (will look like):
   ```
   https://abc123def456-8000.proxy.runpod.net
   ```

3. Test the API by visiting:
   ```
   https://YOUR-POD-ID-8000.proxy.runpod.net/health
   ```
   You should see JSON response with system status.

---

## Part 2: GitHub Upload

### Step 2.1: Create GitHub Repository
1. Go to https://github.com
2. Click **"New repository"**
3. **Repository name**: `athen-ai-medical-system`
4. Set to **Public** (for CDN access)
5. Click **"Create repository"**

### Step 2.2: Upload Files
Upload these files to your repository:

**Essential Files:**
- `main_script_orgid.py` - Main API server
- `requirements.txt` - Python dependencies
- `athen-ai-client.js` - JavaScript client library
- `frontend-integration.html` - Demo interface
- `runpod-automation-integration.js` - Automation tools

**Documentation:**
- `README.md` - Project overview
- `INTEGRATION_GUIDE.md` - Integration instructions
- `DEPLOYMENT_CHECKLIST.md` - Deployment verification

### Step 2.3: Verify Upload
1. Check that all files are visible in your repository
2. Test CDN access by visiting:
   ```
   https://cdn.jsdelivr.net/gh/YOUR-USERNAME/YOUR-REPO@main/athen-ai-client.js
   ```
3. You should see the JavaScript code

---

## Part 3: Frontend Integration

### Step 3.1: Prepare Your Information
Before starting, collect:
- **GitHub Username**: `YOUR-USERNAME`
- **Repository Name**: `YOUR-REPO`
- **RunPod URL**: `https://YOUR-POD-ID-8000.proxy.runpod.net`
- **Organization ID**: `YOUR-HOSPITAL-NAME` (e.g., "emory", "mayo_clinic")

### Step 3.2: Add RunPod Script to Your Website
Open your main HTML file (usually `index.html`) and add this **before the closing `</body>` tag**:

```html
<!-- Athen.ai Medical AI Integration -->
<script src="https://cdn.jsdelivr.net/gh/YOUR-USERNAME/YOUR-REPO@main/athen-ai-client.js"></script>

<script>
console.log('Initializing Athen.ai Medical AI...');

// Initialize RunPod connection
const medicalAI = new AthenAIClient('https://YOUR-POD-ID-8000.proxy.runpod.net', {
    defaultOrgId: 'YOUR-HOSPITAL-NAME',
    debug: true
});

// Test connection when page loads
window.addEventListener('load', async () => {
    try {
        const health = await medicalAI.testConnection();
        console.log('Medical AI System Status:', health);
        
        // Update UI to show connection status
        updateConnectionStatus(health.success);
        
    } catch (error) {
        console.error('Medical AI connection failed:', error);
        updateConnectionStatus(false);
    }
});

// Function to show connection status
function updateConnectionStatus(connected) {
    let statusElement = document.getElementById('ai-status');
    if (!statusElement) {
        statusElement = document.createElement('div');
        statusElement.id = 'ai-status';
        statusElement.style.cssText = `
            position: fixed; 
            top: 10px; 
            right: 10px; 
            padding: 10px 15px; 
            border-radius: 5px; 
            z-index: 1000; 
            font-weight: bold;
        `;
        document.body.appendChild(statusElement);
    }
    
    if (connected) {
        statusElement.innerHTML = '🟢 Medical AI Online';
        statusElement.style.backgroundColor = '#28a745';
        statusElement.style.color = 'white';
    } else {
        statusElement.innerHTML = '🔴 Medical AI Offline';
        statusElement.style.backgroundColor = '#dc3545';
        statusElement.style.color = 'white';
    }
}
</script>
```

**Remember to replace:**
- `YOUR-USERNAME` → Your GitHub username
- `YOUR-REPO` → Your repository name
- `YOUR-POD-ID` → Your RunPod pod ID
- `YOUR-HOSPITAL-NAME` → Your organization identifier

### Step 3.3: Update Your Chat Function
Find your existing chat function (usually in a `.js` file) and replace it with:

```javascript
// Enhanced medical AI chat function
async function askQuestion(question) {
    try {
        // Show loading indicator
        showLoadingIndicator(true);
        
        // Query the medical AI system
        const result = await medicalAI.query(question, {
            orgId: 'YOUR-HOSPITAL-NAME',
            k: 3  // Number of context documents
        });
        
        // Hide loading indicator
        showLoadingIndicator(false);
        
        // Display the AI response
        displayMedicalResponse(result.answer, result.sources, result.confidence);
        
        return result;
        
    } catch (error) {
        console.error('Medical AI query failed:', error);
        showLoadingIndicator(false);
        
        // Show error message
        displayErrorMessage('The medical AI system is temporarily unavailable. Please try again.');
    }
}

// Function to show/hide loading indicator
function showLoadingIndicator(show) {
    let loader = document.getElementById('medical-ai-loader');
    if (!loader) {
        loader = document.createElement('div');
        loader.id = 'medical-ai-loader';
        loader.innerHTML = '🧠 Processing medical query...';
        loader.style.cssText = `
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(0,0,0,0.8);
            color: white;
            padding: 20px;
            border-radius: 10px;
            z-index: 1001;
            display: none;
        `;
        document.body.appendChild(loader);
    }
    
    loader.style.display = show ? 'block' : 'none';
}

// Enhanced response display with medical context
function displayMedicalResponse(answer, sources, confidence) {
    // Use your existing chat display function, or create one
    displayChatMessage(answer, 'ai');
    
    // Optionally show sources and confidence
    if (sources && sources.length > 0) {
        const sourceText = `\n\nSources: ${sources.join(', ')}`;
        displayChatMessage(sourceText, 'sources');
    }
    
    if (confidence) {
        console.log(`Response confidence: ${confidence}`);
    }
}

// Error message display
function displayErrorMessage(message) {
    displayChatMessage(message, 'error');
}
```

### Step 3.4: Add Medical Document Upload Feature
Add this HTML where you want the upload functionality:

```html
<!-- Medical Document Upload Section -->
<div class="medical-upload-section" style="
    margin: 20px 0; 
    padding: 20px; 
    border: 2px dashed #007bff; 
    border-radius: 10px; 
    background: #f8f9fa;
">
    <h3>📚 Upload Medical Training Documents</h3>
    <p>Upload ZIP files containing medical documents to train your AI assistant on your specific content.</p>
    
    <div style="margin: 15px 0;">
        <label for="org-input" style="display: block; margin-bottom: 5px; font-weight: bold;">
            🏥 Organization/Hospital ID:
        </label>
        <input 
            type="text" 
            id="org-input" 
            placeholder="e.g., emory, mayo_clinic, your_hospital" 
            value="YOUR-HOSPITAL-NAME"
            style="width: 300px; padding: 8px; border: 1px solid #ddd; border-radius: 4px;"
        >
    </div>
    
    <div style="margin: 15px 0;">
        <label for="medical-file-input" style="display: block; margin-bottom: 5px; font-weight: bold;">
            📁 Select Medical Documents (ZIP format):
        </label>
        <input 
            type="file" 
            id="medical-file-input" 
            accept=".zip"
            style="margin-bottom: 10px;"
        >
        <small style="display: block; color: #666;">
            Supported: PDF, DOCX, TXT files in ZIP format. Max size: 200MB
        </small>
    </div>
    
    <button 
        onclick="uploadMedicalDocuments()" 
        style="
            background: #007bff; 
            color: white; 
            padding: 12px 24px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            font-size: 16px;
        "
        onmouseover="this.style.background='#0056b3'" 
        onmouseout="this.style.background='#007bff'"
    >
        🚀 Upload & Train Medical AI
    </button>
    
    <div id="upload-progress" style="margin-top: 15px; min-height: 20px;"></div>
</div>
```

Add this JavaScript function:

```javascript
// Medical document upload function
async function uploadMedicalDocuments() {
    const fileInput = document.getElementById('medical-file-input');
    const orgInput = document.getElementById('org-input');
    const progressDiv = document.getElementById('upload-progress');
    
    // Validation
    if (!fileInput.files[0]) {
        alert('Please select a ZIP file containing your medical documents');
        return;
    }
    
    if (!orgInput.value.trim()) {
        alert('Please enter your organization/hospital ID');
        return;
    }
    
    // Show progress
    progressDiv.innerHTML = `
        <div style="color: #007bff; font-weight: bold;">
            ⏳ Uploading and processing medical documents...
            <div style="margin-top: 10px;">
                This may take several minutes depending on file size.
            </div>
        </div>
    `;
    
    try {
        // Upload using the medical AI client
        const result = await medicalAI.uploadTrainingData(
            fileInput.files[0], 
            orgInput.value.trim()
        );
        
        // Show success message
        progressDiv.innerHTML = `
            <div style="color: #28a745; font-weight: bold;">
                ✅ Success! Medical AI training complete.
                <div style="margin-top: 10px;">
                    📊 Processed: ${result.processed_files} files<br>
                    📚 Total documents: ${result.total_chunks} chunks<br>
                    🎯 Ready for medical queries!
                </div>
            </div>
        `;
        
        // Clear the file input
        fileInput.value = '';
        
        // Enable chat if it was disabled
        enableMedicalChat();
        
    } catch (error) {
        console.error('Medical document upload failed:', error);
        
        progressDiv.innerHTML = `
            <div style="color: #dc3545; font-weight: bold;">
                ❌ Upload failed: ${error.message}
                <div style="margin-top: 10px;">
                    Please check your files and try again.
                </div>
            </div>
        `;
    }
}

// Function to enable chat after successful upload
function enableMedicalChat() {
    // Adapt this to your existing chat interface
    const chatInput = document.getElementById('chat-input');
    if (chatInput) {
        chatInput.disabled = false;
        chatInput.placeholder = 'Ask medical questions about your uploaded content...';
    }
}
```

### Step 3.5: Add Advanced Fine-tuning (Optional)
For organizations wanting to improve AI performance with their specific data:

```html
<!-- Advanced AI Fine-tuning Section -->
<div class="fine-tuning-section" style="
    margin: 20px 0; 
    padding: 20px; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    color: white; 
    border-radius: 10px;
">
    <h3>🧠 Advanced: Fine-tune Your Medical AI</h3>
    <p>Improve AI performance by fine-tuning the model on your specific medical content.</p>
    
    <button 
        onclick="startMedicalFineTuning()" 
        style="
            background: #28a745; 
            color: white; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px;
        "
    >
        🔥 Start Fine-tuning
    </button>
    
    <button 
        onclick="evaluateMedicalAI()" 
        style="
            background: #ffc107; 
            color: #212529; 
            padding: 10px 20px; 
            border: none; 
            border-radius: 5px; 
            cursor: pointer; 
            margin: 5px;
        "
    >
        📊 Evaluate Performance
    </button>
    
    <div id="fine-tuning-status" style="margin-top: 15px; min-height: 20px;"></div>
</div>
```

```javascript
// Advanced fine-tuning functions
async function startMedicalFineTuning() {
    const statusDiv = document.getElementById('fine-tuning-status');
    const orgId = document.getElementById('org-input').value.trim() || 'YOUR-HOSPITAL-NAME';
    
    statusDiv.innerHTML = '🧠 Initializing fine-tuning process...';
    
    try {
        const result = await medicalAI.fineTuneModel(orgId, {
            epochs: 3,              // Training cycles
            learning_rate: 2e-4,    // Learning speed
            batch_size: 1,          // Processing batch size
            max_seq_length: 2048    // Maximum text length
        });
        
        statusDiv.innerHTML = `
            ✅ Fine-tuning started successfully!<br>
            📊 Status: ${result.status}<br>
            ⏱️ Estimated time: ${result.estimated_time_minutes} minutes<br>
            🔄 You can continue using the system while training runs in background.
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `❌ Fine-tuning failed: ${error.message}`;
    }
}

async function evaluateMedicalAI() {
    const statusDiv = document.getElementById('fine-tuning-status');
    const orgId = document.getElementById('org-input').value.trim() || 'YOUR-HOSPITAL-NAME';
    
    statusDiv.innerHTML = '📊 Evaluating medical AI performance...';
    
    try {
        const evaluation = await medicalAI.evaluateModel(orgId);
        
        statusDiv.innerHTML = `
            📊 Medical AI Evaluation Results:<br>
            🎯 Average Confidence: ${evaluation.summary.avg_overlap_score.toFixed(2)}<br>
            📝 Total Evaluations: ${evaluation.summary.total_evaluations}<br>
            ⚠️ Quality Issues Detected: ${evaluation.summary.hallucination_count}<br>
            💡 Recommendation: ${evaluation.summary.hallucination_count === 0 ? 'Excellent performance!' : 'Consider fine-tuning for better accuracy'}
        `;
        
    } catch (error) {
        statusDiv.innerHTML = `❌ Evaluation failed: ${error.message}`;
    }
}
```

---

## Part 4: Testing & Validation

### Step 4.1: Basic Connection Test
1. **Open your website** in a browser
2. **Open browser console** (F12 → Console tab)
3. **Look for connection messages**:
   ```
   Initializing Athen.ai Medical AI...
   Medical AI System Status: {success: true, ...}
   ```
4. **Check status indicator** (green dot in top-right)

### Step 4.2: API Health Check
Run this in browser console:
```javascript
medicalAI.testConnection()
    .then(result => console.log('Health Check:', result))
    .catch(error => console.error('Health Check Failed:', error));
```

Expected response:
```json
{
  "success": true,
  "message": "Athen.ai Medical RAG API is running",
  "gpu_info": "NVIDIA GeForce RTX 4090",
  "models_loaded": ["language_model", "embedding_model", "medical_nlp"]
}
```

### Step 4.3: Document Upload Test
1. **Create a test ZIP file** with sample medical content:
   ```
   test_medical.zip
   ├── appendectomy_procedure.txt
   ├── cardiac_anatomy.pdf
   └── emergency_protocols.docx
   ```

2. **Use the upload interface** on your website
3. **Monitor upload progress** 
4. **Verify successful processing**

### Step 4.4: Medical Query Test
Try these test queries:
```
"What are the steps in an appendectomy?"
"Describe the blood supply to the heart"
"What are the emergency protocols for cardiac arrest?"
```

Expected response format:
```
Based on the uploaded medical documents, an appendectomy involves the following key steps:

1. Patient preparation and anesthesia
2. Surgical site preparation and draping
3. Incision (either open or laparoscopic approach)
4. Identification and isolation of the appendix
5. Ligation of the appendiceal artery
6. Removal of the appendix
7. Closure of the appendiceal stump
8. Irrigation and inspection
9. Closure of incisions

Sources: appendectomy_procedure.txt, surgical_techniques.pdf
```

### Step 4.5: Fine-tuning Test (Optional)
1. **Upload substantial medical content** (10+ documents)
2. **Test baseline performance** with complex medical queries
3. **Start fine-tuning process**
4. **Wait for completion** (15-30 minutes)
5. **Test improved performance** with same queries
6. **Compare results** before/after fine-tuning

---

## Part 5: Advanced Features

### Step 5.1: Multi-Organization Setup
For multiple hospitals/departments:

```javascript
// Initialize multiple organization clients
const clients = {
    cardiology: new AthenAIClient(runpodURL, {defaultOrgId: 'cardiology'}),
    surgery: new AthenAIClient(runpodURL, {defaultOrgId: 'surgery'}),
    emergency: new AthenAIClient(runpodURL, {defaultOrgId: 'emergency'})
};

// Switch between departments
function switchDepartment(dept) {
    window.currentAI = clients[dept];
    updateUIForDepartment(dept);
}
```

### Step 5.2: Custom UI Themes
Medical-themed styling:

```css
/* Medical theme CSS */
.medical-theme {
    --primary-color: #0066cc;
    --secondary-color: #28a745;
    --accent-color: #17a2b8;
    --text-color: #333;
    --bg-color: #f8f9fa;
}

.medical-chat-container {
    font-family: 'Segoe UI', Arial, sans-serif;
    max-width: 800px;
    margin: 0 auto;
    background: var(--bg-color);
    border-radius: 15px;
    box-shadow: 0 5px 20px rgba(0,0,0,0.1);
}

.medical-chat-header {
    background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
    color: white;
    padding: 20px;
    border-radius: 15px 15px 0 0;
}

.ai-message {
    background: #e3f2fd;
    border-left: 4px solid var(--primary-color);
    padding: 15px;
    margin: 10px 0;
    border-radius: 0 10px 10px 10px;
}

.user-message {
    background: #f1f8e9;
    border-left: 4px solid var(--secondary-color);
    padding: 15px;
    margin: 10px 0;
    border-radius: 10px 0 10px 10px;
}
```

### Step 5.3: Integration with EMR Systems
For healthcare organizations with existing Electronic Medical Records:

```javascript
// EMR integration example
class EMRIntegration {
    constructor(emrAPI, medicalAI) {
        this.emrAPI = emrAPI;
        this.medicalAI = medicalAI;
    }
    
    async queryWithPatientContext(question, patientId) {
        // Get patient context from EMR
        const patientData = await this.emrAPI.getPatientSummary(patientId);
        
        // Enhanced query with patient context
        const contextualQuery = `
            Patient Context: ${patientData.summary}
            Question: ${question}
        `;
        
        return await this.medicalAI.query(contextualQuery, {
            orgId: 'emr_integration',
            includePatientContext: true
        });
    }
}
```

### Step 5.4: Automated Training Pipeline
Set up automatic retraining:

```javascript
// Automated training scheduler
class MedicalAITrainingScheduler {
    constructor(medicalAI, schedule = 'weekly') {
        this.medicalAI = medicalAI;
        this.schedule = schedule;
        this.setupSchedule();
    }
    
    setupSchedule() {
        const intervals = {
            daily: 24 * 60 * 60 * 1000,
            weekly: 7 * 24 * 60 * 60 * 1000,
            monthly: 30 * 24 * 60 * 60 * 1000
        };
        
        setInterval(() => {
            this.performScheduledTraining();
        }, intervals[this.schedule]);
    }
    
    async performScheduledTraining() {
        console.log('Starting scheduled medical AI training...');
        
        try {
            // Get list of organizations
            const orgs = await this.medicalAI.getOrganizations();
            
            // Train each organization
            for (const org of orgs) {
                await this.medicalAI.fineTuneModel(org.id, {
                    epochs: 2,
                    learning_rate: 1e-4
                });
            }
            
            console.log('Scheduled training completed successfully');
            
        } catch (error) {
            console.error('Scheduled training failed:', error);
        }
    }
}
```

---

## Troubleshooting

### Common Issues & Solutions

#### RunPod Connection Issues
**Problem**: "Connection failed" or timeout errors
**Solutions**:
1. Verify your RunPod is running (check RunPod dashboard)
2. Ensure port 8000 is exposed
3. Check the pod URL format: `https://POD-ID-8000.proxy.runpod.net`
4. Wait for models to fully load (check `/health` endpoint)

#### Upload Failures
**Problem**: "Upload failed" or processing errors
**Solutions**:
1. Ensure files are in ZIP format
2. Check file size limits (200MB max recommended)
3. Verify organization ID is set
4. Check RunPod has sufficient disk space

#### Poor AI Responses
**Problem**: Irrelevant or low-quality answers
**Solutions**:
1. Upload more relevant medical documents
2. Use fine-tuning to improve performance
3. Adjust the `k` parameter (number of context documents)
4. Ensure documents are text-readable (not scanned images)

#### Memory Issues
**Problem**: "CUDA out of memory" errors
**Solutions**:
1. Reduce batch size in fine-tuning
2. Use smaller document chunks
3. Restart the RunPod to clear memory
4. Consider upgrading to higher VRAM GPU

#### Integration Issues
**Problem**: JavaScript errors or function not found
**Solutions**:
1. Check CDN URL is correct
2. Verify all placeholders are replaced
3. Check browser console for specific errors
4. Ensure script loads before calling functions

### Performance Optimization

#### GPU Optimization
```python
# Add to main_script_orgid.py for better performance
import torch
torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

# Optimize memory usage
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
```

#### Response Speed
```javascript
// Client-side caching for faster responses
class CachedMedicalAI {
    constructor(baseClient) {
        this.client = baseClient;
        this.cache = new Map();
    }
    
    async query(question, options) {
        const cacheKey = `${question}_${JSON.stringify(options)}`;
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }
        
        const result = await this.client.query(question, options);
        this.cache.set(cacheKey, result);
        
        return result;
    }
}
```

---

## Appendix

### A. Supported File Formats
- **PDF**: Medical textbooks, research papers, clinical guidelines
- **DOCX/DOC**: Procedure manuals, protocols, documentation
- **TXT**: Plain text medical notes, protocols
- **Images in PDF**: OCR-processed automatically

### B. Organization ID Guidelines
- Use lowercase, no spaces: `mayo_clinic` not `Mayo Clinic`
- Be descriptive: `cardiology_dept` not `cd`
- Keep consistent across all uploads
- Examples: `emory_hospital`, `johns_hopkins`, `medical_center_east`

### C. Fine-tuning Parameters
- **epochs**: 1-5 (more = better learning, longer time)
- **learning_rate**: 1e-5 to 5e-4 (smaller = safer, slower)
- **batch_size**: 1-4 (larger = faster, more memory)
- **max_seq_length**: 1024-4096 (longer = more context, more memory)

### D. API Endpoints Reference
```
GET  /health                    - System status and GPU info
GET  /orgs                      - List organizations with data
POST /upload                    - Upload training documents
POST /query                     - Ask medical questions
POST /fine-tune                 - Start fine-tuning process
POST /evaluate                  - Evaluate model performance
GET  /organizations/{org_id}    - Get org-specific info
```

### E. JavaScript Client Methods
```javascript
// Core methods
await client.testConnection()
await client.uploadTrainingData(file, orgId)
await client.query(question, {orgId, k})
await client.fineTuneModel(orgId, options)
await client.evaluateModel(orgId)

// Utility methods
await client.getOrganizations()
await client.getOrgInfo(orgId)
await client.getSystemHealth()
```

### F. Environment Variables
For advanced RunPod configuration:
```bash
# Memory optimization
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Model cache location
TRANSFORMERS_CACHE=/workspace/model_cache

# API configuration
API_PORT=8000
CORS_ORIGINS=*

# Logging level
LOG_LEVEL=INFO
```

---

## Support & Contact

**Documentation**: https://github.com/your-repo/medical-ai-system
**Issues**: https://github.com/your-repo/medical-ai-system/issues
**Email**: support@athen.ai

**Created by**: Athen.ai Development Team
**Version**: 1.0
**Last Updated**: January 2025

---

*This guide provides complete setup instructions for deploying a production-ready medical AI system. Follow each step carefully and refer to the troubleshooting section if you encounter any issues.*
