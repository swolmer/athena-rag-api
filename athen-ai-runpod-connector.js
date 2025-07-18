/**
 * Athen.ai RunPod Integration - Complete API Connection
 * Connects your website directly to your RunPod medical AI API
 */

// ========================================
// 🔧 CONFIGURATION - UPDATE YOUR RUNPOD URL HERE
// ========================================

// Your specific RunPod URL
const RUNPOD_API_URL = "https://7fw9i838ml6e89-19123-2kbevou0smub53klr189.proxy.runpod.net";

// Default settings
const DEFAULT_CONFIG = {
    orgId: 'your_hospital_name',
    maxRetries: 3,
    timeout: 30000,
    debug: true
};

// ========================================
// 🧠 MAIN API CONNECTION CLASS
// ========================================

class AthenAIRunPodConnector {
    constructor(apiUrl = RUNPOD_API_URL, config = {}) {
        this.apiUrl = apiUrl.replace(/\/$/, ''); // Remove trailing slash
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.isHealthy = false;
        
        // Auto-initialize
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Athen.ai RunPod Connection...');
        await this.checkHealth();
    }

    // ========================================
    // 🏥 HEALTH CHECK
    // ========================================
    
    async checkHealth() {
        try {
            console.log('🔍 Checking RunPod API health...');
            
            const response = await fetch(`${this.apiUrl}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                timeout: this.config.timeout
            });

            if (!response.ok) {
                throw new Error(`Health check failed: ${response.status}`);
            }

            const health = await response.json();
            this.isHealthy = true;
            
            console.log('✅ RunPod API is healthy:', health);
            this.displayConnectionStatus(true, health);
            
            return health;

        } catch (error) {
            console.error('❌ RunPod API health check failed:', error);
            this.isHealthy = false;
            this.displayConnectionStatus(false, { error: error.message });
            throw error;
        }
    }

    // ========================================
    // 📤 FILE UPLOAD TO RUNPOD
    // ========================================
    
    async uploadTrainingData(file, orgId = this.config.orgId, onProgress = null) {
        try {
            console.log(`📤 Uploading training data for org: ${orgId}`);
            
            if (onProgress) onProgress({ step: 'Preparing upload', progress: 10 });

            // Validate file
            if (!file) {
                throw new Error('No file provided');
            }

            if (!file.name.endsWith('.zip')) {
                throw new Error('Only ZIP files are supported');
            }

            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('org_id', orgId);

            if (onProgress) onProgress({ step: 'Uploading to RunPod', progress: 30 });

            // Upload to RunPod
            const response = await fetch(`${this.apiUrl}/upload`, {
                method: 'POST',
                body: formData,
                // Don't set Content-Type header - let browser set it with boundary
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Upload failed: ${response.status}`);
            }

            const result = await response.json();
            
            if (onProgress) onProgress({ step: 'Upload complete', progress: 100 });
            
            console.log('✅ Upload successful:', result);
            return result;

        } catch (error) {
            console.error('❌ Upload failed:', error);
            if (onProgress) onProgress({ step: 'Upload failed', progress: 0, error: error.message });
            throw error;
        }
    }

    // ========================================
    // 💬 QUERY THE MEDICAL AI
    // ========================================
    
    async query(question, orgId = this.config.orgId, k = 3) {
        try {
            console.log(`💬 Querying medical AI: "${question}"`);

            const requestBody = {
                question: question,
                org_id: orgId,
                k: k
            };

            const response = await fetch(`${this.apiUrl}/query`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Query failed: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ Query successful:', result);
            
            return result;

        } catch (error) {
            console.error('❌ Query failed:', error);
            throw error;
        }
    }

    // ========================================
    // 🎨 UI HELPERS
    // ========================================
    
    displayConnectionStatus(isConnected, data = {}) {
        // Remove existing status if any
        const existing = document.getElementById('runpod-status');
        if (existing) existing.remove();

        // Create status indicator
        const statusDiv = document.createElement('div');
        statusDiv.id = 'runpod-status';
        statusDiv.style.cssText = `
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 10px 15px;
            border-radius: 8px;
            z-index: 10000;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 14px;
            font-weight: 600;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.3s ease;
        `;

        if (isConnected) {
            statusDiv.innerHTML = `
                🟢 Medical AI Online
                ${data.gpu_name ? `<br><small style="font-weight: normal; opacity: 0.8;">GPU: ${data.gpu_name}</small>` : ''}
            `;
            statusDiv.style.backgroundColor = '#10b981';
            statusDiv.style.color = 'white';
        } else {
            statusDiv.innerHTML = `🔴 Medical AI Offline`;
            statusDiv.style.backgroundColor = '#ef4444';
            statusDiv.style.color = 'white';
        }

        document.body.appendChild(statusDiv);

        // Auto-hide after 5 seconds if connected
        if (isConnected) {
            setTimeout(() => {
                if (statusDiv && statusDiv.parentNode) {
                    statusDiv.style.opacity = '0.6';
                }
            }, 5000);
        }
    }
}

// ========================================
// 🚀 AUTO-INITIALIZATION
// ========================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeAthenAI);
} else {
    initializeAthenAI();
}

function initializeAthenAI() {
    console.log('🏥 Initializing Athen.ai Medical AI Integration...');
    
    // Create global instances
    window.athenAI = {
        connector: new AthenAIRunPodConnector()
    };

    console.log('✅ Athen.ai integration ready!');
    console.log('📋 Available methods:');
    console.log('   - window.athenAI.connector.checkHealth()');
    console.log('   - window.athenAI.connector.query("your question")');
    console.log('   - window.athenAI.connector.uploadTrainingData(file)');
}

// ========================================
// 🎯 QUICK TEST FUNCTIONS
// ========================================

// Test connection
window.testRunPodConnection = async function() {
    try {
        const health = await window.athenAI.connector.checkHealth();
        console.log('✅ Connection test successful:', health);
        return health;
    } catch (error) {
        console.error('❌ Connection test failed:', error);
        return null;
    }
};

// Test query
window.testMedicalQuery = async function(question = "What is appendectomy?") {
    try {
        const result = await window.athenAI.connector.query(question);
        console.log('✅ Query test successful:', result);
        return result;
    } catch (error) {
        console.error('❌ Query test failed:', error);
        return null;
    }
};
