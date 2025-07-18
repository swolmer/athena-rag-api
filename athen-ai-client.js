/**
 * 🚀 Athen.ai RunPod API Client Library
 * Easy integration with existing websites for medical RAG queries
 */

class AthenAIClient {
    constructor(apiBaseUrl, options = {}) {
        this.apiBaseUrl = apiBaseUrl.endsWith('/') ? apiBaseUrl.slice(0, -1) : apiBaseUrl;
        this.defaultOrgId = options.defaultOrgId || null;
        this.defaultK = options.defaultK || 3;
        this.debug = options.debug || false;
    }

    // Log debug messages
    _log(message, data = null) {
        if (this.debug) {
            console.log(`[AthenAI] ${message}`, data || '');
        }
    }

    // Test API connection
    async testConnection() {
        try {
            this._log('Testing API connection...');
            const response = await fetch(`${this.apiBaseUrl}/health`);
            const data = await response.json();
            
            if (response.ok) {
                this._log('API connection successful', data);
                return {
                    success: true,
                    status: 'online',
                    gpu: data.gpu_name || data.gpu_info || 'GPU Ready',
                    cudaAvailable: data.cuda_available,
                    data: data
                };
            } else {
                throw new Error('Health check failed');
            }
        } catch (error) {
            this._log('API connection failed', error);
            return {
                success: false,
                status: 'offline',
                error: error.message
            };
        }
    }

    // Upload training data for an organization
    async uploadTrainingData(file, orgId = null) {
        const targetOrgId = orgId || this.defaultOrgId;
        
        if (!targetOrgId) {
            throw new Error('Organization ID is required');
        }
        
        if (!file || !file.name.endsWith('.zip')) {
            throw new Error('A ZIP file is required');
        }

        try {
            this._log(`Uploading training data for org: ${targetOrgId}`);
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('org_id', targetOrgId);
            
            const response = await fetch(`${this.apiBaseUrl}/upload`, {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this._log('Upload successful', data);
                return {
                    success: true,
                    orgId: data.org_id,
                    filename: data.filename,
                    message: data.message,
                    processedFiles: data.processed_files || 'Multiple files'
                };
            } else {
                throw new Error(data.detail || 'Upload failed');
            }
        } catch (error) {
            this._log('Upload failed', error);
            throw error;
        }
    }

    // Query the RAG system
    async query(question, options = {}) {
        const orgId = options.orgId || this.defaultOrgId;
        const k = options.k || this.defaultK;
        
        if (!orgId) {
            throw new Error('Organization ID is required');
        }
        
        if (!question.trim()) {
            throw new Error('Question cannot be empty');
        }

        try {
            this._log(`Querying org ${orgId}: ${question}`);
            
            const response = await fetch(`${this.apiBaseUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question.trim(),
                    org_id: orgId,
                    k: k
                })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                this._log('Query successful', data);
                return {
                    success: true,
                    answer: data.answer,
                    contextChunks: data.context_chunks,
                    orgId: data.org_id,
                    question: question
                };
            } else {
                throw new Error(data.detail || 'Query failed');
            }
        } catch (error) {
            this._log('Query failed', error);
            throw error;
        }
    }

    // List available organizations
    async getOrganizations() {
        try {
            this._log('Fetching organizations...');
            
            const response = await fetch(`${this.apiBaseUrl}/orgs`);
            const data = await response.json();
            
            if (response.ok) {
                this._log('Organizations fetched', data);
                return {
                    success: true,
                    organizations: data.organizations || []
                };
            } else {
                throw new Error('Failed to fetch organizations');
            }
        } catch (error) {
            this._log('Failed to fetch organizations', error);
            throw error;
        }
    }

    // Check if an organization has training data
    async hasTrainingData(orgId) {
        try {
            const result = await this.getOrganizations();
            const org = result.organizations.find(o => o.org_id === orgId);
            return org ? org.has_training_data : false;
        } catch (error) {
            return false;
        }
    }
}

// UI Helper Class for easy integration
class AthenAIUI {
    constructor(client, containerId) {
        this.client = client;
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Container with ID '${containerId}' not found`);
        }
    }

    // Create a complete chat interface
    createChatInterface(options = {}) {
        const orgId = options.orgId || this.client.defaultOrgId;
        
        this.container.innerHTML = `
            <div class="athen-chat-container" style="max-width: 800px; margin: 0 auto; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: white; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); overflow: hidden;">
                <div class="athen-chat-header" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; text-align: center;">
                    <h3 style="margin: 0 0 5px 0; font-size: 1.5em;">🏥 Athen.ai Medical Assistant</h3>
                    <small style="opacity: 0.9;">Powered by RunPod RTX 4090 • Hermes-2-Pro-Mistral-7B</small>
                </div>
                
                <div class="athen-upload-section" style="padding: 20px; background: #f8f9fa; border-bottom: 1px solid #e9ecef;">
                    <div style="margin-bottom: 15px;">
                        <label style="display: block; margin-bottom: 5px; font-weight: 600; color: #495057;">📤 Upload Training Data (ZIP file):</label>
                        <input type="file" id="athen-file-upload" accept=".zip" style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px; margin-bottom: 10px;">
                        <button onclick="athenUIInstance.uploadFile()" style="background: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: 600;">Upload Training Data</button>
                    </div>
                    <div>
                        <label style="display: block; margin-bottom: 5px; font-weight: 600; color: #495057;">🏥 Organization ID:</label>
                        <input type="text" id="athen-org-display" value="${orgId || 'Not set'}" readonly style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 5px; background: #e9ecef; color: #6c757d;">
                    </div>
                </div>
                
                <div class="athen-chat-messages" id="athen-messages" style="height: 400px; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 15px;">
                    <div class="athen-message athen-bot" style="align-self: flex-start;">
                        <div style="background: #e3f2fd; padding: 15px; border-radius: 15px; max-width: 80%; border: 1px solid #bbdefb;">
                            👋 <strong>Welcome to Athen.ai!</strong><br><br>
                            📋 <strong>Getting started:</strong><br>
                            1. Upload a ZIP file with medical documents above<br>
                            2. Ask questions about surgical procedures, anatomy, or medical techniques<br>
                            3. Get AI-powered answers based on your training data<br><br>
                            💡 <strong>Example questions:</strong><br>
                            • "What are the steps in laparoscopic appendectomy?"<br>
                            • "Describe the blood supply to the pancreas"<br>
                            • "What are the contraindications for robotic surgery?"
                        </div>
                    </div>
                </div>
                
                <div class="athen-chat-input" style="padding: 20px; background: white; border-top: 1px solid #e9ecef; display: flex; gap: 10px;">
                    <input type="text" id="athen-question-input" placeholder="Ask a medical question..." 
                           style="flex: 1; padding: 12px 15px; border: 1px solid #ddd; border-radius: 25px; outline: none; font-size: 16px;">
                    <button onclick="athenUIInstance.sendMessage()" id="athen-send-btn" style="background: #667eea; color: white; border: none; padding: 12px 25px; border-radius: 25px; cursor: pointer; font-weight: 600; white-space: nowrap;">Send</button>
                </div>
            </div>
        `;

        // Add event listener for Enter key
        document.getElementById('athen-question-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendMessage();
            }
        });

        // Store reference for onclick handlers
        window.athenUIInstance = this;
    }

    // Send message in chat interface
    async sendMessage() {
        const input = document.getElementById('athen-question-input');
        const sendBtn = document.getElementById('athen-send-btn');
        const question = input.value.trim();
        
        if (!question) return;

        // Clear input and disable button
        input.value = '';
        sendBtn.disabled = true;
        sendBtn.textContent = 'Sending...';

        // Add user message
        this.addMessage(question, 'user');

        // Add thinking message
        const thinkingId = this.addMessage('🤔 Analyzing your question...', 'bot');

        try {
            const result = await this.client.query(question);
            
            // Remove thinking message
            document.getElementById(thinkingId).remove();
            
            // Add bot response
            this.addMessage(result.answer, 'bot', result.contextChunks);
            
        } catch (error) {
            // Remove thinking message
            document.getElementById(thinkingId).remove();
            
            // Add error message
            if (error.message.includes('Organization ID')) {
                this.addMessage(`⚠️ Please upload training data first before asking questions.`, 'bot', null, true);
            } else {
                this.addMessage(`❌ Error: ${error.message}`, 'bot', null, true);
            }
        } finally {
            // Re-enable button
            sendBtn.disabled = false;
            sendBtn.textContent = 'Send';
        }
    }

    // Upload file in chat interface
    async uploadFile() {
        const fileInput = document.getElementById('athen-file-upload');
        const file = fileInput.files[0];
        
        if (!file) {
            alert('Please select a ZIP file to upload');
            return;
        }

        try {
            this.addMessage(`📤 Uploading ${file.name}...`, 'bot');
            
            const result = await this.client.uploadTrainingData(file);
            
            this.addMessage(`✅ <strong>Upload successful!</strong><br>
                            📁 File: ${result.filename}<br>
                            🏥 Organization: ${result.orgId}<br>
                            📊 Processed: ${result.processedFiles}<br><br>
                            You can now ask questions about the uploaded materials!`, 'bot');
            
            // Clear file input
            fileInput.value = '';
            
        } catch (error) {
            this.addMessage(`❌ <strong>Upload failed:</strong> ${error.message}<br><br>
                            💡 <strong>Tips:</strong><br>
                            • Make sure the file is a ZIP archive<br>
                            • Check that it contains PDF, DOCX, or text files<br>
                            • Ensure the file size is under 100MB`, 'bot', null, true);
        }
    }

    // Add message to chat
    addMessage(content, sender, contextChunks = null, isError = false) {
        const messagesContainer = document.getElementById('athen-messages');
        const messageId = 'msg-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
        
        const messageDiv = document.createElement('div');
        messageDiv.id = messageId;
        messageDiv.className = `athen-message athen-${sender}`;
        messageDiv.style.alignSelf = sender === 'user' ? 'flex-end' : 'flex-start';
        
        const bubbleColor = sender === 'user' ? '#667eea' : (isError ? '#dc3545' : '#e3f2fd');
        const textColor = sender === 'user' ? 'white' : (isError ? 'white' : '#333');
        const borderColor = sender === 'user' ? '#667eea' : (isError ? '#dc3545' : '#bbdefb');
        
        let contextHtml = '';
        if (contextChunks && contextChunks.length > 0) {
            contextHtml = `
                <div style="margin-top: 10px; padding: 10px; background: rgba(255,255,255,0.1); border-radius: 8px; font-size: 12px;">
                    📚 <strong>Based on ${contextChunks.length} source document${contextChunks.length > 1 ? 's' : ''}</strong>
                </div>
            `;
        }
        
        messageDiv.innerHTML = `
            <div style="background: ${bubbleColor}; color: ${textColor}; padding: 15px; border-radius: 15px; max-width: 80%; border: 1px solid ${borderColor}; line-height: 1.5;">
                ${content}
                ${contextHtml}
            </div>
        `;
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageId;
    }
}

// Example usage:
/*
// Initialize the client with your RunPod URL
const athenAI = new AthenAIClient('https://your-pod-id-8000.proxy.runpod.net', {
    defaultOrgId: 'emory',
    debug: true
});

// Test connection
athenAI.testConnection().then(result => {
    console.log('Connection test:', result);
});

// Create chat interface in a div with id="chat-container"
const chatUI = new AthenAIUI(athenAI, 'chat-container');
chatUI.createChatInterface();

// Or use the API directly
athenAI.query('What are the key steps in appendectomy?').then(result => {
    console.log('Answer:', result.answer);
    console.log('Context:', result.contextChunks);
});
*/

// Export for Node.js if available
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AthenAIClient, AthenAIUI };
}
