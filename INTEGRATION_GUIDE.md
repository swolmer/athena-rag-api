# 🚀 Athen.ai Frontend Integration Guide

This guide shows how to connect your existing website to the RunPod-deployed Athen.ai RAG API.

## 📋 Quick Setup Checklist

1. ✅ **Get your RunPod URL**: Your API is running at `https://[your-pod-id]-8000.proxy.runpod.net`
2. ✅ **Choose integration method**: Standalone page OR existing website integration
3. ✅ **Test the connection**: Verify your API endpoints are working
4. ✅ **Upload training data**: Add medical documents for your organization
5. ✅ **Start querying**: Ask medical questions and get AI-powered answers

## 🌐 Available API Endpoints

Your RunPod API provides these endpoints:

```javascript
GET  /health           // Check API status and GPU info (RTX 4090)
GET  /orgs             // List all organizations with training data
POST /upload           // Upload ZIP file with training materials
POST /query            // Ask questions and get RAG answers
POST /fine-tune        // Fine-tune model for specific organization
POST /evaluate         // Evaluate model performance with test questions
```

### Model Information
- **Language Model**: NousResearch Hermes-2-Pro-Mistral-7B (Medical optimized)
- **Embedding Model**: all-MiniLM-L6-v2 (Running on CUDA)
- **GPU**: NVIDIA RTX 4090 
- **Framework**: FastAPI with CORS enabled
- **Fine-tuning**: ✅ Supported with custom training data

## 🔧 Integration Methods

### Method 1: Standalone HTML Page (Easiest)
Use `frontend-integration.html` - a complete ready-to-use interface:

1. Open `frontend-integration.html` in any browser
2. Enter your RunPod URL: `https://your-pod-id-8000.proxy.runpod.net`
3. Upload a ZIP file with medical documents
4. Start asking questions!

### Method 2: JavaScript Client Library
Use `athen-ai-client.js` to integrate with existing websites:

```html
<!-- Add to your existing website -->
<script src="athen-ai-client.js"></script>
<div id="medical-chat"></div>

<script>
// Initialize with your RunPod URL
const athenAI = new AthenAIClient('https://your-pod-id-8000.proxy.runpod.net', {
    defaultOrgId: 'your_hospital_name',
    debug: true
});

// Create chat interface
const chatUI = new AthenAIUI(athenAI, 'medical-chat');
chatUI.createChatInterface();
</script>
```

### Method 3: Custom API Integration
Build your own interface using fetch requests:

```javascript
// Example: Custom query function
async function askMedicalQuestion(question, orgId) {
    const response = await fetch('https://your-pod-id-8000.proxy.runpod.net/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            question: question,
            org_id: orgId,
            k: 3
        })
    });
    
    const result = await response.json();
    return result.answer;
}
```

## 📤 Upload Training Data

### Step 1: Prepare Your Data
- Create a ZIP file with medical documents (PDF, DOCX, images)
- Organize by specialty or department if needed
- Example structure:
  ```
  medical_training.zip
  ├── surgery_procedures.pdf
  ├── anatomy_guide.docx
  └── surgical_techniques.pdf
  ```

### Step 2: Upload via API
```javascript
const formData = new FormData();
formData.append('file', zipFile);
formData.append('org_id', 'your_hospital_name');

fetch('https://your-pod-id-8000.proxy.runpod.net/upload', {
    method: 'POST',
    body: formData
}).then(response => response.json())
  .then(data => console.log('Upload success:', data));
```

## 💬 Query Examples

Once training data is uploaded, you can ask questions like:

```javascript
// Surgical procedures
await athenAI.query("What are the key steps in laparoscopic appendectomy?");

// Anatomy questions  
await athenAI.query("Describe the blood supply to the pancreas");

// Technical procedures
await athenAI.query("What are the contraindications for robotic surgery?");

// Complications and management
await athenAI.query("How do you manage postoperative bleeding after thyroidectomy?");
```

## 🧠 Fine-tuning Your Model

Once you've uploaded training data, you can fine-tune the model for better performance:

### Basic Fine-tuning
```javascript
const client = new AthenAIClient('https://your-pod-id-8000.proxy.runpod.net');

// Start fine-tuning
const fineTuneResult = await client.fineTuneModel('your_hospital_name', {
    epochs: 3,
    learning_rate: 2e-4,
    batch_size: 1
});

console.log('Fine-tuning status:', fineTuneResult.status);
console.log('Estimated completion:', fineTuneResult.estimated_time_minutes);
```

### Evaluate Model Performance
```javascript
// Check how well your model is performing
const evaluation = await client.evaluateModel('your_hospital_name');

console.log('Average confidence:', evaluation.summary.avg_overlap_score);
console.log('Total evaluations:', evaluation.summary.total_evaluations);
console.log('Hallucinations detected:', evaluation.summary.hallucination_count);

// View detailed results
evaluation.results.forEach(result => {
    console.log(`Question: ${result.question}`);
    console.log(`Confidence: ${result.overlap_score}`);
    console.log(`Answer quality: ${result.is_hallucination ? 'Poor' : 'Good'}`);
});
```

### Manual Fine-tuning via API
```javascript
async function fineTuneModel(orgId, options = {}) {
    const response = await fetch(`${API_URL}/fine-tune`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            org_id: orgId,
            epochs: options.epochs || 3,
            learning_rate: options.learning_rate || 2e-4,
            batch_size: options.batch_size || 1,
            max_seq_length: options.max_seq_length || 2048
        })
    });
    return response.json();
}
```

## 🔗 Integration with Existing Athen.ai Website

If you already have an Athen.ai website, follow these steps:

### 1. Find your existing JavaScript files
Look for files like:
- `script.js`
- `main.js` 
- `app.js`
- Any file with API calls

### 2. Update the API base URL
Replace any existing API URLs with your RunPod URL:

```javascript
// OLD (if you had a local or different API)
const API_URL = 'http://localhost:8000';

// NEW (your RunPod URL)
const API_URL = 'https://your-pod-id-8000.proxy.runpod.net';
```

### 3. Update API endpoints
Make sure your existing functions use the correct endpoints:

```javascript
// Upload function
async function uploadTrainingData(file, orgId) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('org_id', orgId);
    
    const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
    });
    return response.json();
}

// Query function  
async function askQuestion(question, orgId, k = 3) {
    const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, org_id: orgId, k })
    });
    return response.json();
}
```

## 🛠️ Troubleshooting

### Testing Your RunPod Connection

First, verify your API is running correctly:

```javascript
// Test connection and view system info
fetch('https://your-pod-id-8000.proxy.runpod.net/health')
    .then(r => r.json())
    .then(data => {
        console.log('API Status:', data);
        console.log('GPU:', data.gpu_info);
        console.log('Model:', data.model_info);
    });

// Expected response:
// {
//   "status": "healthy",
//   "gpu_info": "NVIDIA GeForce RTX 4090",
//   "model_info": "NousResearch--Hermes-2-Pro-Mistral-7B loaded"
// }
```

### Common Issues:

**1. CORS Errors**
- ✅ Your RunPod API already has CORS enabled
- If you still see CORS errors, check the URL format
- Ensure you're using HTTPS, not HTTP

**2. API Not Responding**
- Check your RunPod instance is running
- Verify the port (8000) is exposed
- Test with: `curl https://your-pod-id-8000.proxy.runpod.net/health`

**3. Upload Failures**
- Ensure ZIP file is under 100MB
- Check organization ID format (no spaces, use underscores)
- Verify the ZIP contains supported files (PDF, DOCX, TXT, images)
- File formats supported: `.pdf`, `.docx`, `.txt`, `.png`, `.jpg`, `.jpeg`

**4. No Training Data Found**
- Upload training materials first before querying
- Check organization ID matches exactly between upload and query
- Use `/orgs` endpoint to see available organizations:
  ```javascript
  fetch('https://your-pod-id-8000.proxy.runpod.net/orgs')
      .then(r => r.json())
      .then(orgs => console.log('Available orgs:', orgs));
  ```

**5. Poor Quality Answers**
- Ensure uploaded documents are relevant to your questions
- Try increasing the `k` parameter (default: 3, try 5-10 for more context)
- Check document quality - scanned PDFs may have OCR issues

## 📱 Mobile-Friendly Integration

The provided interfaces are responsive. For mobile optimization:

```css
@media (max-width: 768px) {
    .athen-chat-container {
        margin: 10px;
        max-width: none;
    }
    
    .athen-chat-messages {
        height: 300px;
    }
}
```

## 🔐 Security Notes

- Your RunPod URL is already secured with HTTPS
- Consider adding authentication if needed for production
- The API includes basic rate limiting
- Uploaded files are stored securely per organization

## 📞 Next Steps

1. **Get your RunPod URL** from your RunPod dashboard
2. **Test the connection** using the health endpoint
3. **Upload training data** for your organization  
4. **Integrate with your website** using one of the methods above
5. **Customize the UI** to match your branding

## 📁 Required Files for GitHub Upload

To deploy this complete medical AI system, upload these files to your repository:

### ✅ Core Integration Files (Frontend)
- `athen-ai-client.js` - JavaScript API client library with fine-tuning support
- `runpod-automation-integration.js` - Automated training pipeline
- `frontend-integration.html` - Standalone demo interface with fine-tuning UI
- `INTEGRATION_GUIDE.md` - This comprehensive documentation

### ✅ Backend Files (RunPod Server)
- `main_script_orgid.py` - Main RunPod API server with fine-tuning endpoints
- `requirements.txt` - Python dependencies (version conflicts fixed)
- `README.md` - Project documentation

### ✅ Documentation Files
- `COMPLETE_SETUP_GUIDE.md` - Comprehensive setup instructions
- `DEPLOYMENT_NOTES.md` - Deployment and troubleshooting guide

### 🔧 Fixed Version Conflicts
The `requirements.txt` has been updated to resolve:
- ✅ **spacy version conflict**: Downgraded from 3.7.4 to 3.6.1 for scispacy compatibility
- ✅ **Fine-tuning dependencies**: Added peft, trl, bitsandbytes, datasets for advanced training
- ✅ **All dependencies verified**: Compatible versions for production deployment

### 📤 GitHub Upload Checklist
Before uploading to GitHub:

1. **Verify RunPod URL**: Update all example URLs in documentation with your actual RunPod endpoint
2. **Test fine-tuning**: Ensure `/fine-tune` and `/evaluate` endpoints work correctly
3. **Organization IDs**: Update example organization IDs in documentation
4. **Security**: Remove any API keys or sensitive information from code
5. **README**: Update README.md with your specific deployment instructions

### 🚀 Ready for Production
Your system now includes:
- ✅ **RAG capabilities** for document-based Q&A
- ✅ **Fine-tuning system** for improved model performance  
- ✅ **Evaluation framework** for quality assurance
- ✅ **Complete automation** from upload to deployment
- ✅ **Organization isolation** for multi-tenant deployment
- ✅ **Version-compatible dependencies** for smooth installation

Need help? The API is already running and ready to use! 🚀
