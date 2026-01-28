class ClassFocusApp {
    constructor() {
        this.videoElement = document.getElementById('videoElement');
        this.captureCanvas = document.getElementById('captureCanvas');
        this.overlayCanvas = document.createElement('canvas');  // For face box overlay
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.emotionDisplay = document.getElementById('emotionDisplay');
        this.statusDisplay = document.getElementById('statusDisplay');
        
        // Stats elements
        this.totalFramesEl = document.getElementById('totalFrames');
        this.attentiveFramesEl = document.getElementById('attentiveFrames');
        this.sessionTimeEl = document.getElementById('sessionTime');
        this.attentionPercentageEl = document.getElementById('attentionPercentage');
        
        // Final results elements
        this.finalResults = document.getElementById('finalResults');
        this.finalPercentageEl = document.getElementById('finalPercentage');
        this.finalAttentiveTimeEl = document.getElementById('finalAttentiveTime');
        this.finalTotalTimeEl = document.getElementById('finalTotalTime');
        this.resultsMessage = document.getElementById('resultsMessage');
        
        // Session data
        this.isAnalyzing = false;
        this.totalFrames = 0;
        this.attentiveFrames = 0;
        this.sessionStartTime = null;
        this.analysisInterval = null;
        this.stream = null;
        
        this.initializeEventListeners();
        this.updateStatusDisplay('initializing', 'Camera not started');
    }
    
    initializeEventListeners() {
        this.startBtn.addEventListener('click', () => this.startCamera());
        this.stopBtn.addEventListener('click', () => this.stopCamera());
    }
    
    async startCamera() {
        try {
            this.updateStatusDisplay('initializing', 'Starting camera...');
            this.emotionDisplay.textContent = 'Initializing camera...';
            
            // Request camera access
            this.stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });
            
            this.videoElement.srcObject = this.stream;
            
            // Wait for video to load
            await new Promise((resolve) => {
                this.videoElement.onloadedmetadata = resolve;
            });
            
            // Set up overlay canvas for face box
            this.setupOverlayCanvas();
            
            // Initialize session
            this.resetSession();
            this.sessionStartTime = Date.now();
            
            // Start analysis
            this.isAnalyzing = true;
            this.startAnalysis();
            
            // Update UI
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;
            this.finalResults.style.display = 'none';
            
            this.updateStatusDisplay('initializing', 'AI analyzing...');
            this.emotionDisplay.textContent = 'AI processing...';
            
        } catch (error) {
            console.error('Error starting camera:', error);
            this.updateStatusDisplay('error', 'Camera access denied');
            this.emotionDisplay.textContent = 'Camera access required';
            
            // User-friendly error message
            const errorMsg = error.name === 'NotAllowedError' 
                ? 'Please allow camera access and try again'
                : 'Camera not available. Please check your device.';
            
            alert(errorMsg);
        }
    }
    
    stopCamera() {
        this.isAnalyzing = false;
        
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
        }
        
        if (this.stream) {
            this.stream.getTracks().forEach(track => track.stop());
            this.stream = null;
        }
        
        this.videoElement.srcObject = null;
        
        // Update UI
        this.startBtn.disabled = false;
        this.stopBtn.disabled = true;
        
        // Show final results
        this.showFinalResults();
        
        this.updateStatusDisplay('initializing', 'Session ended');
        this.emotionDisplay.textContent = 'Session complete';
    }
    
    startAnalysis() {
        // Analyze frame every 1.5 seconds for better performance on low-end hardware
        this.analysisInterval = setInterval(() => {
            if (this.isAnalyzing) {
                this.captureAndAnalyze();
                this.updateSessionTime();
            }
        }, 1500);  // Increased interval for better performance
        
        // Initial analysis after a short delay
        setTimeout(() => {
            if (this.isAnalyzing) {
                this.captureAndAnalyze();
            }
        }, 1000);  // Slightly longer delay for initialization
    }
    
    async captureAndAnalyze() {
        try {
            // Show processing status for user feedback
            //this.emotionDisplay.textContent = 'Processing...';
            let lastEmotion = "Neutral";

            async function analyzeFrame() {
              try {
                const res = await fetch("/analyze", {
                 method: "POST",
                 headers: { "Content-Type": "application/json" },
                 body: JSON.stringify({ image: getFrameBase64() })
                });

                const data = await res.json();

                if (data.dominant) {
                  lastEmotion = data.dominant +
                    (data.intensity ? " " + data.intensity : "");
                  statusText.innerText = lastEmotion;
                }
              } catch (e) {
                // do NOTHING → keep last emotion
              }
            }

            
            // Set canvas dimensions to match video
            const canvas = this.captureCanvas;
            const context = canvas.getContext('2d');
            
            canvas.width = this.videoElement.videoWidth;
            canvas.height = this.videoElement.videoHeight;
            
            // Draw current video frame to canvas
            context.drawImage(this.videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert to base64 JPEG with lower quality for faster processing
            const imageData = canvas.toDataURL('image/jpeg', 0.6);
            
            // Send to backend for analysis with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout
            
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
                signal: controller.signal
            });
            
            clearTimeout(timeoutId);
            
            const result = await response.json();
            
            if (response.ok) {
                this.processAnalysisResult(result);
            } else {
                console.error('Analysis error:', result.error);
                this.updateStatusDisplay('error', 'Analysis failed');
                this.emotionDisplay.textContent = 'AI processing error';
            }
            
        } catch (error) {
            if (error.name === 'AbortError') {
                console.error('Analysis timeout');
                this.updateStatusDisplay('error', 'Processing timeout');
                this.emotionDisplay.textContent = 'Processing timeout - trying again...';
            } else {
                console.error('Capture and analysis error:', error);
                this.updateStatusDisplay('error', 'Connection error');
                this.emotionDisplay.textContent = 'Connection lost';
            }
        }
    }
    
    processAnalysisResult(result) {
        const { dominant, intensity, attentive, probability, face_box, sleeping } = result;
        
        // Update frame counts
        this.totalFrames++;
        if (attentive) {
            this.attentiveFrames++;
        }
        
        // Draw face box if detected
        this.drawFaceBox(face_box);
        
        // Update emotion display with enhanced formatting
        let emotionText;
        const probPercent = Math.round((probability || 0) * 100);
        const emotionCapitalized = this.capitalizeFirst(dominant);
        
         if (!intensity) {
             emotionText = emotionCapitalized;
        }  else {
            emotionText = `${emotionCapitalized} (${intensity})`;
        }

        this.emotionDisplay.textContent = emotionText;

        
        this.emotionDisplay.textContent = emotionText;
        
        // Update status display with enhanced styling - handle sleeping
        let statusText, statusClass;
        if (sleeping) {
            statusText = 'Not Attentive - Sleeping';
            statusClass = 'sleeping';
        } else {
            statusText = attentive ? 'Attentive' : 'Not Attentive';
            statusClass = attentive ? 'attentive' : 'not-attentive';
        }
        this.updateStatusDisplay(statusClass, statusText);
        
        // Update stats
        this.updateStats();
        
        // Add visual feedback for engagement
        this.updateEngagementFeedback(attentive && !sleeping);
    }
    
    setupOverlayCanvas() {
        // Position overlay canvas on top of video
        const videoContainer = document.querySelector('.video-container');
        this.overlayCanvas.style.position = 'absolute';
        this.overlayCanvas.style.top = '0';
        this.overlayCanvas.style.left = '0';
        this.overlayCanvas.style.pointerEvents = 'none';
        this.overlayCanvas.style.zIndex = '10';
        
        // Match video dimensions
        this.overlayCanvas.width = this.videoElement.videoWidth;
        this.overlayCanvas.height = this.videoElement.videoHeight;
        this.overlayCanvas.style.width = '100%';
        this.overlayCanvas.style.height = '100%';
        
        videoContainer.appendChild(this.overlayCanvas);
    }
    
    drawFaceBox(faceBox) {
        const ctx = this.overlayCanvas.getContext('2d');
        
        // Clear previous drawings
        ctx.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        
        if (faceBox) {
            const [x, y, w, h] = faceBox;
            
            // Scale coordinates to canvas size
            const scaleX = this.overlayCanvas.width / this.videoElement.videoWidth;
            const scaleY = this.overlayCanvas.height / this.videoElement.videoHeight;
            
            const scaledX = x * scaleX;
            const scaledY = y * scaleY;
            const scaledW = w * scaleX;
            const scaledH = h * scaleY;
            
            // Draw face bounding box
            ctx.strokeStyle = '#021206ff';  // Green color like phone cameras
            ctx.lineWidth = 3;
            ctx.setLineDash([]);
            
            // Draw rounded rectangle
            this.drawRoundedRect(ctx, scaledX, scaledY, scaledW, scaledH, 8);
            
            // Add corner markers for a more phone-camera look
            this.drawCornerMarkers(ctx, scaledX, scaledY, scaledW, scaledH);
        }
    }
    
    drawRoundedRect(ctx, x, y, width, height, radius) {
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + width - radius, y);
        ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
        ctx.lineTo(x + width, y + height - radius);
        ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
        ctx.lineTo(x + radius, y + height);
        ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.closePath();
        ctx.stroke();
    }
    
    drawCornerMarkers(ctx, x, y, width, height) {
        const cornerSize = 20;
        ctx.strokeStyle = '#021206ff';
        ctx.lineWidth = 4;
        
        // Top-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + cornerSize);
        ctx.lineTo(x, y);
        ctx.lineTo(x + cornerSize, y);
        ctx.stroke();
        
        // Top-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerSize, y);
        ctx.lineTo(x + width, y);
        ctx.lineTo(x + width, y + cornerSize);
        ctx.stroke();
        
        // Bottom-left corner
        ctx.beginPath();
        ctx.moveTo(x, y + height - cornerSize);
        ctx.lineTo(x, y + height);
        ctx.lineTo(x + cornerSize, y + height);
        ctx.stroke();
        
        // Bottom-right corner
        ctx.beginPath();
        ctx.moveTo(x + width - cornerSize, y + height);
        ctx.lineTo(x + width, y + height);
        ctx.lineTo(x + width, y + height - cornerSize);
        ctx.stroke();
    }

    updateStatusDisplay(type, text) {
        const statusSpan = this.statusDisplay.querySelector('span');
        if (statusSpan) {
            statusSpan.textContent = text;
        } else {
            this.statusDisplay.textContent = text;
        }
        
        // Reset classes
        this.statusDisplay.className = 'status-badge';
        
        // Add appropriate status class
        switch (type) {
            case 'attentive':
                this.statusDisplay.classList.add('status-attentive');
                break;
            case 'not-attentive':
                this.statusDisplay.classList.add('status-not-attentive');
                break;
            case 'sleeping':
                this.statusDisplay.classList.add('status-sleeping');
                break;
            case 'error':
                this.statusDisplay.classList.add('status-error');
                break;
            default:
                this.statusDisplay.classList.add('status-initializing');
        }
    }
    
    updateStats() {
        this.totalFramesEl.textContent = this.totalFrames;
        this.attentiveFramesEl.textContent = this.attentiveFrames;
        
        const percentage = this.totalFrames > 0 
            ? Math.round((this.attentiveFrames / this.totalFrames) * 100)
            : 0;
        
        this.attentionPercentageEl.textContent = `${percentage}%`;
        
        // Enhanced color coding with better thresholds
        this.attentionPercentageEl.className = 'stat-value primary';
        if (percentage >= 80) {
            this.attentionPercentageEl.classList.add('text-success');
        } else if (percentage >= 60) {
            this.attentionPercentageEl.classList.add('text-warning');
        } else if (percentage < 40) {
            this.attentionPercentageEl.classList.add('text-danger');
        }
    }
    
    updateSessionTime() {
        if (this.sessionStartTime) {
            const elapsed = Math.floor((Date.now() - this.sessionStartTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            
            if (minutes > 0) {
                this.sessionTimeEl.textContent = `${minutes}m ${seconds}s`;
            } else {
                this.sessionTimeEl.textContent = `${seconds}s`;
            }
        }
    }
    
    updateEngagementFeedback(isAttentive) {
        // Add subtle visual feedback to the video container
        const videoContainer = document.querySelector('.video-container');
        if (videoContainer) {
            videoContainer.classList.remove('engagement-positive', 'engagement-negative');
            if (isAttentive) {
                videoContainer.classList.add('engagement-positive');
            } else {
                videoContainer.classList.add('engagement-negative');
            }
            
            // Remove the class after animation
            setTimeout(() => {
                videoContainer.classList.remove('engagement-positive', 'engagement-negative');
            }, 1000);
        }
    }
    
    showFinalResults() {
        const totalTime = this.sessionStartTime 
            ? Math.floor((Date.now() - this.sessionStartTime) / 1000)
            : 0;
        
        const attentiveTime = this.attentiveFrames; // Approximate seconds (1 frame per second)
        const percentage = this.totalFrames > 0 
            ? Math.round((this.attentiveFrames / this.totalFrames) * 100)
            : 0;
        
        // Update final results
        this.finalPercentageEl.textContent = `${percentage}%`;
        this.finalAttentiveTimeEl.textContent = attentiveTime;
        this.finalTotalTimeEl.textContent = totalTime;
        
        // Enhanced final score styling
        if (percentage >= 80) {
            this.finalPercentageEl.className = 'score-value text-success';
            this.resultsMessage.textContent = 'Excellent focus! You maintained great attention throughout the session.';
        } else if (percentage >= 60) {
            this.finalPercentageEl.className = 'score-value text-warning';
            this.resultsMessage.textContent = 'Good job! You showed solid attention with room for improvement.';
        } else if (percentage >= 40) {
            this.finalPercentageEl.className = 'score-value text-info';
            this.resultsMessage.textContent = 'Fair attention level. Try to minimize distractions in your next session.';
        } else {
            this.finalPercentageEl.className = 'score-value text-danger';
            this.resultsMessage.textContent = 'Your attention was low this session. Consider taking breaks and reducing distractions.';
        }
        
        this.finalResults.style.display = 'block';
        
        // Scroll to results
        setTimeout(() => {
            this.finalResults.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 300);
    }
    
    resetSession() {
        this.totalFrames = 0;
        this.attentiveFrames = 0;
        this.sessionStartTime = null;
        this.updateStats();
        this.sessionTimeEl.textContent = '0s';
    }
    
    capitalizeFirst(str) {
        return str.charAt(0).toUpperCase() + str.slice(1);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ClassFocusApp();
    
    // Add some CSS for engagement feedback
    const style = document.createElement('style');
    style.textContent = `
        .video-container.engagement-positive {
            border: 3px solid rgba(56, 239, 125, 0.8) !important;
            box-shadow: 0 0 30px rgba(56, 239, 125, 0.3) !important;
            transition: all 0.3s ease;
        }
        
        .video-container.engagement-negative {
            border: 3px solid rgba(252, 70, 107, 0.8) !important;
            box-shadow: 0 0 30px rgba(252, 70, 107, 0.3) !important;
            transition: all 0.3s ease;
        }
    `;
    document.head.appendChild(style);
});