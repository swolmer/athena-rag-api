/**
 * 🤖 RunPod Automation Integration for Athen.ai
 * 
 * Advanced automation pipeline that handles:
 * 1. Automatic file upload processing
 * 2. Training pipeline management
 * 3. Website integration automation
 * 4. Real-time status monitoring
 */

class RunPodAutoTrainer {
    constructor(runpodUrl, options = {}) {
        this.runpodUrl = runpodUrl.replace(/\/$/, '');
        this.orgId = options.orgId || 'default';
        this.autoRetry = options.autoRetry !== false;
        this.maxRetries = options.maxRetries || 3;
        this.debug = options.debug || false;
        this.statusCallbacks = [];
    }

    // Add status update callback
    onStatusUpdate(callback) {
        this.statusCallbacks.push(callback);
    }

    // Emit status updates
    _emitStatus(status, data = {}) {
        const statusUpdate = {
            timestamp: new Date().toISOString(),
            status: status,
            ...data
        };
        
        if (this.debug) {
            console.log('[RunPodAutoTrainer]', statusUpdate);
        }

        this.statusCallbacks.forEach(callback => {
            try {
                callback(statusUpdate);
            } catch (error) {
                console.error('Status callback error:', error);
            }
        });
    }

    // Automated training pipeline
    async trainFromFiles(files, orgId = null) {
        const targetOrgId = orgId || this.orgId;
        
        try {
            this._emitStatus('pipeline_started', {
                orgId: targetOrgId,
                fileCount: files.length
            });

            // Step 1: Validate files
            this._emitStatus('validating_files');
            await this._validateFiles(files);

            // Step 2: Create ZIP archive
            this._emitStatus('creating_archive');
            const zipFile = await this._createZipArchive(files);

            // Step 3: Upload to RunPod
            this._emitStatus('uploading_to_runpod');
            const uploadResult = await this._uploadWithRetry(zipFile, targetOrgId);

            // Step 4: Wait for processing
            this._emitStatus('processing_training_data');
            await this._waitForProcessing(targetOrgId);

            // Step 5: Validate training completion
            this._emitStatus('validating_training');
            const trainingStatus = await this._validateTraining(targetOrgId);

            this._emitStatus('pipeline_completed', {
                orgId: targetOrgId,
                uploadResult: uploadResult,
                trainingStatus: trainingStatus
            });

            return {
                success: true,
                orgId: targetOrgId,
                uploadResult: uploadResult,
                trainingStatus: trainingStatus
            };

        } catch (error) {
            this._emitStatus('pipeline_failed', {
                error: error.message,
                orgId: targetOrgId
            });
            throw error;
        }
    }

    // Validate files before processing
    async _validateFiles(files) {
        const validExtensions = ['.pdf', '.docx', '.txt', '.doc', '.rtf'];
        const maxFileSize = 50 * 1024 * 1024; // 50MB per file
        const maxTotalSize = 200 * 1024 * 1024; // 200MB total

        let totalSize = 0;
        const invalidFiles = [];

        for (const file of files) {
            totalSize += file.size;
            
            if (file.size > maxFileSize) {
                invalidFiles.push(`${file.name} is too large (max 50MB)`);
            }

            const extension = '.' + file.name.split('.').pop().toLowerCase();
            if (!validExtensions.includes(extension)) {
                invalidFiles.push(`${file.name} has unsupported format`);
            }
        }

        if (totalSize > maxTotalSize) {
            throw new Error(`Total file size (${Math.round(totalSize/1024/1024)}MB) exceeds limit (200MB)`);
        }

        if (invalidFiles.length > 0) {
            throw new Error(`Invalid files:\n${invalidFiles.join('\n')}`);
        }

        this._emitStatus('files_validated', {
            fileCount: files.length,
            totalSizeMB: Math.round(totalSize / 1024 / 1024)
        });
    }

    // Create ZIP archive from files
    async _createZipArchive(files) {
        // This would require a ZIP library like JSZip in a real implementation
        // For now, we'll assume files are already in ZIP format or use FormData
        
        if (files.length === 1 && files[0].name.endsWith('.zip')) {
            return files[0];
        }

        // In a real implementation, you'd use JSZip:
        /*
        const JSZip = require('jszip');
        const zip = new JSZip();
        
        for (const file of files) {
            zip.file(file.name, file);
        }
        
        return await zip.generateAsync({type:"blob"});
        */
        
        throw new Error('Multiple file upload requires ZIP format. Please ZIP your files first.');
    }

    // Upload with automatic retry
    async _uploadWithRetry(zipFile, orgId, retryCount = 0) {
        try {
            const formData = new FormData();
            formData.append('file', zipFile);
            formData.append('org_id', orgId);

            const response = await fetch(`${this.runpodUrl}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }

            const result = await response.json();
            
            this._emitStatus('upload_completed', {
                orgId: orgId,
                filename: result.filename,
                processedFiles: result.processed_files
            });

            return result;

        } catch (error) {
            if (this.autoRetry && retryCount < this.maxRetries) {
                this._emitStatus('upload_retry', {
                    retryCount: retryCount + 1,
                    maxRetries: this.maxRetries,
                    error: error.message
                });

                // Exponential backoff
                const delay = Math.pow(2, retryCount) * 1000;
                await new Promise(resolve => setTimeout(resolve, delay));

                return this._uploadWithRetry(zipFile, orgId, retryCount + 1);
            }

            throw error;
        }
    }

    // Wait for processing to complete
    async _waitForProcessing(orgId, maxWait = 60000) {
        const startTime = Date.now();
        const pollInterval = 2000; // 2 seconds

        while (Date.now() - startTime < maxWait) {
            try {
                const response = await fetch(`${this.runpodUrl}/orgs`);
                const data = await response.json();
                
                const org = data.organizations?.find(o => o.org_id === orgId);
                
                if (org && org.has_training_data) {
                    this._emitStatus('processing_completed', { orgId: orgId });
                    return true;
                }

                this._emitStatus('processing_waiting', {
                    orgId: orgId,
                    elapsed: Date.now() - startTime,
                    maxWait: maxWait
                });

                await new Promise(resolve => setTimeout(resolve, pollInterval));

            } catch (error) {
                this._emitStatus('processing_check_error', {
                    error: error.message
                });
                
                await new Promise(resolve => setTimeout(resolve, pollInterval));
            }
        }

        throw new Error(`Processing timeout after ${maxWait/1000} seconds`);
    }

    // Validate training completion
    async _validateTraining(orgId) {
        try {
            // Test query to ensure the system is working
            const testQuery = "What is the main topic of the uploaded documents?";
            
            const response = await fetch(`${this.runpodUrl}/query`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    question: testQuery,
                    org_id: orgId,
                    k: 1
                })
            });

            if (!response.ok) {
                throw new Error(`Validation query failed: ${response.status}`);
            }

            const result = await response.json();
            
            if (!result.answer || result.answer.length < 10) {
                throw new Error('Training validation failed: insufficient response');
            }

            this._emitStatus('training_validated', {
                orgId: orgId,
                testQuery: testQuery,
                answerLength: result.answer.length
            });

            return {
                valid: true,
                testQuery: testQuery,
                testAnswer: result.answer
            };

        } catch (error) {
            this._emitStatus('training_validation_failed', {
                error: error.message
            });
            throw error;
        }
    }

    // Get training status
    async getTrainingStatus(orgId = null) {
        const targetOrgId = orgId || this.orgId;
        
        try {
            const response = await fetch(`${this.runpodUrl}/orgs`);
            const data = await response.json();
            
            const org = data.organizations?.find(o => o.org_id === targetOrgId);
            
            return {
                orgId: targetOrgId,
                hasTrainingData: org ? org.has_training_data : false,
                isReady: org ? org.has_training_data : false
            };

        } catch (error) {
            return {
                orgId: targetOrgId,
                hasTrainingData: false,
                isReady: false,
                error: error.message
            };
        }
    }
}

/**
 * 🌐 Website Integration Automation
 * 
 * Automatically integrates Athen.ai into existing websites
 */
class AthenAIWebsiteIntegration {
    constructor(options = {}) {
        this.autoInit = options.autoInit !== false;
        this.containerId = options.containerId || 'athen-ai-container';
        this.theme = options.theme || 'light';
        this.debug = options.debug || false;
    }

    // Automatically detect and integrate with common website frameworks
    async autoIntegrate(runpodUrl, orgId) {
        try {
            // Detect framework
            const framework = this._detectFramework();
            
            if (this.debug) {
                console.log('Detected framework:', framework);
            }

            // Create integration based on framework
            switch (framework) {
                case 'wordpress':
                    return this._integrateWordPress(runpodUrl, orgId);
                case 'shopify':
                    return this._integrateShopify(runpodUrl, orgId);
                case 'squarespace':
                    return this._integrateSquarespace(runpodUrl, orgId);
                case 'react':
                    return this._integrateReact(runpodUrl, orgId);
                case 'vanilla':
                default:
                    return this._integrateVanilla(runpodUrl, orgId);
            }

        } catch (error) {
            console.error('Auto-integration failed:', error);
            // Fallback to vanilla integration
            return this._integrateVanilla(runpodUrl, orgId);
        }
    }

    // Detect website framework
    _detectFramework() {
        // WordPress detection
        if (document.body.classList.contains('wordpress') || 
            document.querySelector('meta[name="generator"][content*="WordPress"]')) {
            return 'wordpress';
        }

        // Shopify detection
        if (window.Shopify || document.querySelector('script[src*="shopify"]')) {
            return 'shopify';
        }

        // Squarespace detection
        if (document.body.classList.contains('squarespace') ||
            document.querySelector('script[src*="squarespace"]')) {
            return 'squarespace';
        }

        // React detection
        if (window.React || document.querySelector('[data-reactroot]')) {
            return 'react';
        }

        return 'vanilla';
    }

    // WordPress integration
    _integrateWordPress(runpodUrl, orgId) {
        // Create WordPress shortcode equivalent
        const shortcode = `
            <div id="${this.containerId}"></div>
            <script>
                (function() {
                    ${this._getIntegrationScript(runpodUrl, orgId)}
                })();
            </script>
        `;

        // Try to add to content area
        const content = document.querySelector('.entry-content, .content, main');
        if (content) {
            content.insertAdjacentHTML('beforeend', shortcode);
            return { success: true, framework: 'wordpress' };
        }

        return this._integrateVanilla(runpodUrl, orgId);
    }

    // Vanilla HTML integration
    _integrateVanilla(runpodUrl, orgId) {
        // Create container if it doesn't exist
        let container = document.getElementById(this.containerId);
        if (!container) {
            container = document.createElement('div');
            container.id = this.containerId;
            document.body.appendChild(container);
        }

        // Load and execute integration
        const script = document.createElement('script');
        script.textContent = this._getIntegrationScript(runpodUrl, orgId);
        document.head.appendChild(script);

        return { success: true, framework: 'vanilla' };
    }

    // Get integration script
    _getIntegrationScript(runpodUrl, orgId) {
        return `
            // Load Athen.ai client if not already loaded
            if (typeof AthenAIClient === 'undefined') {
                const script = document.createElement('script');
                script.src = 'https://cdn.jsdelivr.net/gh/swolmer/athena-rag-api@main/athen-ai-client.js';
                script.onload = initAthenAI;
                document.head.appendChild(script);
            } else {
                initAthenAI();
            }

            function initAthenAI() {
                const client = new AthenAIClient('${runpodUrl}', {
                    defaultOrgId: '${orgId}',
                    debug: ${this.debug}
                });

                const ui = new AthenAIUI(client, '${this.containerId}');
                ui.createChatInterface({
                    theme: '${this.theme}'
                });

                // Test connection
                client.testConnection().then(result => {
                    console.log('Athen.ai connected:', result.success);
                }).catch(error => {
                    console.error('Athen.ai connection failed:', error);
                });
            }
        `;
    }

    // Create floating chat button
    createFloatingChat(runpodUrl, orgId, options = {}) {
        const position = options.position || 'bottom-right';
        const color = options.color || '#667eea';
        
        const button = document.createElement('div');
        button.innerHTML = '💬';
        button.style.cssText = `
            position: fixed;
            ${position.includes('bottom') ? 'bottom: 20px;' : 'top: 20px;'}
            ${position.includes('right') ? 'right: 20px;' : 'left: 20px;'}
            width: 60px;
            height: 60px;
            background: ${color};
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            font-size: 24px;
            transition: all 0.3s ease;
        `;

        button.addEventListener('mouseenter', () => {
            button.style.transform = 'scale(1.1)';
        });

        button.addEventListener('mouseleave', () => {
            button.style.transform = 'scale(1)';
        });

        button.addEventListener('click', () => {
            this._toggleChatModal(runpodUrl, orgId);
        });

        document.body.appendChild(button);
        return button;
    }

    // Toggle chat modal
    _toggleChatModal(runpodUrl, orgId) {
        let modal = document.getElementById('athen-ai-modal');
        
        if (modal) {
            modal.style.display = modal.style.display === 'none' ? 'block' : 'none';
            return;
        }

        // Create modal
        modal = document.createElement('div');
        modal.id = 'athen-ai-modal';
        modal.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.5);
            z-index: 10001;
            display: flex;
            align-items: center;
            justify-content: center;
        `;

        const modalContent = document.createElement('div');
        modalContent.style.cssText = `
            background: white;
            border-radius: 15px;
            width: 90%;
            max-width: 600px;
            height: 80%;
            max-height: 700px;
            position: relative;
            overflow: hidden;
        `;

        const closeBtn = document.createElement('button');
        closeBtn.innerHTML = '×';
        closeBtn.style.cssText = `
            position: absolute;
            top: 10px;
            right: 15px;
            background: none;
            border: none;
            font-size: 24px;
            cursor: pointer;
            z-index: 10002;
        `;

        closeBtn.addEventListener('click', () => {
            modal.style.display = 'none';
        });

        modalContent.appendChild(closeBtn);

        const chatContainer = document.createElement('div');
        chatContainer.id = 'athen-ai-modal-chat';
        modalContent.appendChild(chatContainer);
        modal.appendChild(modalContent);
        document.body.appendChild(modal);

        // Initialize chat in modal
        const client = new AthenAIClient(runpodUrl, {
            defaultOrgId: orgId,
            debug: this.debug
        });

        const ui = new AthenAIUI(client, 'athen-ai-modal-chat');
        ui.createChatInterface();
    }
}

// Export classes
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { RunPodAutoTrainer, AthenAIWebsiteIntegration };
}

// Make available globally
window.RunPodAutoTrainer = RunPodAutoTrainer;
window.AthenAIWebsiteIntegration = AthenAIWebsiteIntegration;
