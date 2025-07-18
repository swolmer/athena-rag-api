# 🚀 Athen.ai RunPod Integration Files

This directory contains all the files needed to integrate your RunPod-deployed Athen.ai API with any website.

## 📁 Files Included

### Core Integration Files
- **`athen-ai-client.js`** - JavaScript API client library for easy website integration
- **`frontend-integration.html`** - Complete standalone demo interface 
- **`runpod-automation-integration.js`** - Advanced automation pipeline for training and deployment
- **`INTEGRATION_README.md`** - This documentation file

## 🚀 Quick Start

### 1. Test Your RunPod API
Open `frontend-integration.html` in any web browser:
1. Enter your RunPod URL: `https://your-pod-id-8000.proxy.runpod.net`
2. Set your organization ID (e.g., `emory`, `hospital_name`)
3. Click "Connect & Test API"
4. Upload a ZIP file with medical documents
5. Start asking medical questions!

### 2. Integrate with Existing Website
Add to your HTML page:
```html
<script src="athen-ai-client.js"></script>
<div id="medical-chat"></div>

<script>
const athenAI = new AthenAIClient('https://your-pod-id-8000.proxy.runpod.net', {
    defaultOrgId: 'your_hospital_name'
});

const chatUI = new AthenAIUI(athenAI, 'medical-chat');
chatUI.createChatInterface();
</script>
```

## 🌐 Your RunPod System

Based on your terminal output, your system is running:

- **🖥️ GPU**: NVIDIA GeForce RTX 4090
- **🧠 Language Model**: NousResearch Hermes-2-Pro-Mistral-7B  
- **🔍 Embedding Model**: all-MiniLM-L6-v2 (CUDA enabled)
- **🌐 API Server**: FastAPI with CORS enabled on port 8000
- **📊 Status**: ✅ All models loaded successfully

## 📋 API Endpoints

Your RunPod provides these endpoints:

```
GET  /health           - Check API status and GPU info
GET  /orgs             - List organizations with training data  
POST /upload           - Upload ZIP file with training materials
POST /query            - Ask questions and get RAG answers
```

## 💻 Example Usage

### Basic API Usage
```javascript
const client = new AthenAIClient('https://your-pod-id-8000.proxy.runpod.net');

// Test connection
const health = await client.testConnection();
console.log('Status:', health.success);

// Upload training data
const file = document.getElementById('file-input').files[0];
await client.uploadTrainingData(file, 'emory');

// Query the system
const result = await client.query('What are the steps in appendectomy?', {
    orgId: 'emory',
    k: 3
});
console.log('Answer:', result.answer);
```

### Advanced Automation
```javascript
const trainer = new RunPodAutoTrainer('https://your-pod-id-8000.proxy.runpod.net', {
    orgId: 'emory',
    debug: true
});

// Monitor training progress
trainer.onStatusUpdate(status => {
    console.log('Training status:', status);
});

// Automated training pipeline
const files = [/* your medical documents */];
await trainer.trainFromFiles(files, 'emory');
```

## 🎨 UI Customization

### Styling the Chat Interface
```css
.athen-chat-container {
    max-width: 800px;
    margin: 20px auto;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}

.athen-chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
}
```

### Creating Floating Chat Button
```javascript
const integration = new AthenAIWebsiteIntegration();
integration.createFloatingChat('https://your-pod-id-8000.proxy.runpod.net', 'emory', {
    position: 'bottom-right',
    color: '#667eea'
});
```

## 🔧 Configuration Options

### AthenAIClient Options
```javascript
const options = {
    defaultOrgId: 'emory',          // Default organization
    defaultK: 3,                    // Number of context chunks
    debug: true                     // Enable debug logging
};
```

### UI Options
```javascript
const uiOptions = {
    orgId: 'emory',                 // Organization ID
    theme: 'light',                 // 'light' or 'dark'
    title: 'Medical Assistant',     // Chat header title
    placeholder: 'Ask a question...' // Input placeholder
};
```

## 📤 File Upload Requirements

### Supported Formats
- ✅ **PDF files** (.pdf) - Medical textbooks, research papers
- ✅ **Word documents** (.docx, .doc) - Procedure manuals  
- ✅ **Text files** (.txt) - Notes and guidelines
- ✅ **ZIP archives** (.zip) - Multiple documents

### File Limits
- 📏 **Max file size**: 50MB per file
- 📊 **Max total size**: 200MB per ZIP
- 📁 **Recommended**: 10-50 documents per organization

### Organization Structure
```
medical_training.zip
├── surgery/
│   ├── appendectomy_procedure.pdf
│   ├── laparoscopic_techniques.docx
│   └── surgical_complications.txt
├── anatomy/
│   ├── cardiac_anatomy.pdf
│   └── vascular_system.docx
└── protocols/
    ├── emergency_procedures.pdf
    └── post_op_care.txt
```

## 🛠️ Troubleshooting

### Connection Issues
```javascript
// Test your RunPod connection
fetch('https://your-pod-id-8000.proxy.runpod.net/health')
    .then(r => r.json())
    .then(data => console.log('API Status:', data))
    .catch(err => console.error('Connection failed:', err));
```

### Common Solutions
- ✅ **CORS enabled**: Your RunPod API already has CORS configured
- 🔒 **HTTPS required**: Ensure your RunPod URL uses HTTPS
- 🔄 **Model loading**: Wait for models to fully load (check /health endpoint)
- 📁 **ZIP format**: Upload files must be in ZIP format

### Error Messages
| Error | Solution |
|-------|----------|
| "Organization ID is required" | Set defaultOrgId or pass orgId parameter |
| "A ZIP file is required" | Upload files in ZIP format only |
| "Connection failed" | Check RunPod URL and ensure pod is running |
| "No training data found" | Upload training materials before querying |

## 🔐 Security Best Practices

### API Security
- 🔒 Your RunPod URL is already HTTPS secured
- 🏥 Organization isolation keeps data separate
- 🔑 Consider adding authentication for production use
- 📊 Built-in rate limiting prevents abuse

### Data Privacy
- 📁 Files are processed per organization
- 🗄️ Training data is isolated by org_id
- 🔒 No data sharing between organizations
- 🚫 No external API calls for processing

## 📞 Support & Resources

### Documentation
- 📚 [GitHub Repository](https://github.com/swolmer/athena-rag-api)
- 🔧 [API Documentation](https://github.com/swolmer/athena-rag-api/blob/main/README.md)
- 💡 [Integration Examples](https://github.com/swolmer/athena-rag-api/examples)

### Community
- 💬 [Issues & Support](https://github.com/swolmer/athena-rag-api/issues)
- 📧 Email: support@athen.ai
- 🌟 Star the repo if this helps you!

---

**🎉 Your RunPod is ready! Start uploading medical documents and asking questions.**
