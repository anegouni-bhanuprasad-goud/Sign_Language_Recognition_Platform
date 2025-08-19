// Global variables
let videoElement;
let outputCanvas;
let canvasCtx;
let holistic;
let camera;
let isRecording = false;
let isPredicting = false;
let frameCount = 0;
let capturedFrames = [];
let totalPredictions = 0;
let lastPredictionTime = 0;
let fpsCounter = 0;
let fpsInterval;

// Global variables for upload mode
let uploadedVideoFile = null;
let currentMode = 'live'; // 'live' or 'upload'


// Start camera (loads canvas and camera feed)
async function startCamera() {
    try {
        // First start the camera
        const stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: 1024,
                height: 1024,
                frameRate: 20
            }
        });

        videoElement.srcObject = stream;
        isRecording = true;

        // Hook up MediaPipe camera utility (but don't start prediction yet)
        camera = new Camera(videoElement, {
            onFrame: async () => {
                if (holistic) {
                    await holistic.send({ image: videoElement });
                }
            },
            width: 1024,
            height: 1024
        });

        camera.start();

        // Update button states - camera is now running
        document.getElementById('startCameraBtn').disabled = true;
        document.getElementById('startCameraBtn').style.display = 'none';
        document.getElementById('startPredictionBtn').disabled = false;
        document.getElementById('startPredictionBtn').style.display = 'inline-flex';
        document.getElementById('stopCameraBtn').disabled = false;
        document.getElementById('stopCameraBtn').style.display = 'inline-flex';

        // Reset counters
        frameCount = 0;
        
        startFPSCounter();
        hideError();
        
        // Update frame counter to show camera is ready
        document.getElementById('frameCounter').textContent = 'Camera ready - Click Start Prediction to begin';
        
    } catch (error) {
        showError('Failed to access camera. Please ensure camera permissions are granted.');
        console.error('Camera error:', error);
    }
}

// Stop camera (stops both camera and prediction)
function stopCamera() {
    if (camera) {
        camera.stop();
        camera = null;
    }
    if (videoElement && videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    if (videoElement) {
        videoElement.pause();
        videoElement.removeAttribute('src');
        videoElement.load();
    }
    isRecording = false;
    isPredicting = false;
    frameCount = 0;
    capturedFrames = [];
    // Reset UI controls
    const startCameraBtn = document.getElementById('startCameraBtn');
    if (startCameraBtn) {
        startCameraBtn.disabled = false;
        startCameraBtn.style.display = 'inline-flex';
    }
    document.getElementById('startPredictionBtn').disabled = true;
    document.getElementById('startPredictionBtn').style.display = 'none';
    document.getElementById('stopPredictionBtn').disabled = true;
    document.getElementById('stopPredictionBtn').style.display = 'none';
    document.getElementById('stopCameraBtn').disabled = true;
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('predictionStatus').style.display = 'none';
    // Clear the canvas so last frame is not shown
    if (canvasCtx && outputCanvas) {
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    }
    updateFrameCounter();
    stopFPSCounter();
}


// MediaPipe Holistic configuration
function initializeMediaPipe() {
    videoElement = document.getElementById('videoElement');
    outputCanvas = document.getElementById('outputCanvas');
    canvasCtx = outputCanvas.getContext('2d');

    // Set canvas size
    outputCanvas.width = 1024;
    outputCanvas.height = 1024;

    holistic = new Holistic({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
        }
    });

    holistic.setOptions({
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5
    });

    holistic.onResults(onResults);
}


// Capture frame data for prediction (up to 60 frames)
function captureFrameData(results) {
    if (capturedFrames.length >= 60) {
        return; // Already have enough frames
    }

    // Build the exact landmark payload (matches temp.json format)
    const frameData = {
        frame_id: capturedFrames.length,
        face_landmarks: results.faceLandmarks
            ? results.faceLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z || 0 }))
            : [],
        pose_landmarks: results.poseLandmarks
            ? results.poseLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z || 0, visibility: lm.visibility || 0 }))
            : [],
        left_hand_landmarks: results.leftHandLandmarks
            ? results.leftHandLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z || 0 }))
            : [],
        right_hand_landmarks: results.rightHandLandmarks
            ? results.rightHandLandmarks.map(lm => ({ x: lm.x, y: lm.y, z: lm.z || 0 }))
            : []
    };

    capturedFrames.push(frameData);

    // Once we've collected exactly 60 frames, trigger a prediction
    if (capturedFrames.length === 60) {
        // Pause further capture while processing the current batch
        isPredicting = false;
        makeLivePrediction();
    }
}



// Handle MediaPipe results: draw both raw image + landmarks
function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);

    // Draw the raw image frame
    canvasCtx.drawImage(results.image, 0, 0, outputCanvas.width, outputCanvas.height);

    // Draw face mesh landmarks
    if (results.faceLandmarks) {
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_TESSELATION, { color: '#C0C0C070', lineWidth: 1 });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYE, { color: '#FF3030' });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_RIGHT_EYEBROW, { color: '#FF3030' });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYE, { color: '#30FF30' });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_LEFT_EYEBROW, { color: '#30FF30' });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_FACE_OVAL, { color: '#E0E0E0' });
        drawConnectors(canvasCtx, results.faceLandmarks, FACEMESH_LIPS, { color: '#E0E0E0' });
    }

    // Draw pose landmarks
    if (results.poseLandmarks) {
        drawConnectors(canvasCtx, results.poseLandmarks, POSE_CONNECTIONS, { color: '#00FF00', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.poseLandmarks, { color: '#FF0000', lineWidth: 2 });
    }

    // Draw left hand landmarks
    if (results.leftHandLandmarks) {
        drawConnectors(canvasCtx, results.leftHandLandmarks, HAND_CONNECTIONS, { color: '#CC0000', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.leftHandLandmarks, { color: '#00FF00', lineWidth: 2 });
    }

    // Draw right hand landmarks
    if (results.rightHandLandmarks) {
        drawConnectors(canvasCtx, results.rightHandLandmarks, HAND_CONNECTIONS, { color: '#0000CC', lineWidth: 4 });
        drawLandmarks(canvasCtx, results.rightHandLandmarks, { color: '#FF0000', lineWidth: 2 });
    }

    canvasCtx.restore();

    // Capture frame data for prediction if toggled ON
    if (isPredicting) {
        captureFrameData(results);
    }

    frameCount++;
    updateFrameCounter();
}



// Start prediction (camera must already be running)
function startPrediction() {
    if (!isRecording) {
        showError('Please start camera first');
        return;
    }

    isPredicting = true;

    // Update button states
    document.getElementById('startPredictionBtn').disabled = true;
    document.getElementById('startPredictionBtn').style.display = 'none';
    document.getElementById('stopPredictionBtn').disabled = false;
    document.getElementById('stopPredictionBtn').style.display = 'inline-flex';
    
    // Show prediction status
    document.getElementById('predictionStatus').style.display = 'inline-flex';
    document.getElementById('predictionSection').classList.add('show');

    // Reset prediction-specific counters
    capturedFrames = [];
    
    hideError();
}



// Stop prediction (keeps camera running)
function stopPrediction() {
    isPredicting = false;
    capturedFrames = [];

    // Update button states - back to camera running state
    document.getElementById('startPredictionBtn').disabled = false;
    document.getElementById('startPredictionBtn').style.display = 'inline-flex';
    document.getElementById('stopPredictionBtn').disabled = true;
    document.getElementById('stopPredictionBtn').style.display = 'none';
    
    // Hide prediction status
    document.getElementById('predictionStatus').style.display = 'none';

    // Update frame counter to show camera is ready
    document.getElementById('frameCounter').textContent = 'Camera ready - Click Start Prediction to begin';
}

// **Make live prediction**: send filtered 30 frames → get result → auto-stop after completion
async function makeLivePrediction() {
    const currentTime = Date.now();
    if (currentTime - lastPredictionTime < 1000) {
        // Enforce ≥1 second between successive requests
        return;
    }
    lastPredictionTime = currentTime;

    // **FILTER OUT ONLY EVEN-INDEXED FRAMES** (0, 2, 4, …, 58) to match temp.json's frame_data
    const filteredFrames = capturedFrames.filter((_, idx) => idx % 2 === 0);
    const predictionData = {
        frame_data: filteredFrames
    };

    try {
        showLoading();
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(predictionData)
        });
        const result = await response.json();
        if (response.ok) {
            showPrediction(result.prediction);
            showPredictedOutputLive(result.prediction);
            totalPredictions++;
            updateStats();
            addToHistory(result.prediction);
            
            // Auto-stop prediction after successful result
            setTimeout(() => {
                stopPrediction();
            }, 1500); // Small delay to show the result before stopping
        } else {
            console.error('Prediction failed:', result);
        }
    } catch (error) {
        console.error('Network error:', error);
    } finally {
        hideLoading();
        // Reset capturedFrames for next potential prediction
        capturedFrames = [];
    }
}

// Show prediction in the “Prediction Section”
function showPrediction(prediction) {
    const match = prediction.match(/Model Predicted : (.+)/);
    const signWord = match ? match[1] : prediction;

    document.getElementById('signWord').textContent = signWord;
    document.getElementById('confidence').textContent = `Prediction made at ${new Date().toLocaleTimeString()}`;
}

// Show predicted output in the new div after prediction (upload mode)
function showPredictedOutput(prediction) {
    const predictedOutputDiv = document.getElementById('predictedOutput');
    if (predictedOutputDiv) {
        const match = prediction.match(/Model Predicted : (.+)/);
        const signWord = match ? match[1] : prediction;
        predictedOutputDiv.textContent = `Predicted Output: ${signWord}`;
        predictedOutputDiv.style.display = 'block';
    }
}

// Show predicted output in the new div after prediction (live mode)
function showPredictedOutputLive(prediction) {
    const predictedOutputDiv = document.getElementById('predictedOutput');
    if (predictedOutputDiv) {
        const match = prediction.match(/Model Predicted : (.+)/);
        const signWord = match ? match[1] : prediction;
        predictedOutputDiv.textContent = `Predicted Output: ${signWord}`;
        predictedOutputDiv.style.display = 'block';
    }
}

// Hide predicted output div
function hidePredictedOutput() {
    const predictedOutputDiv = document.getElementById('predictedOutput');
    if (predictedOutputDiv) {
        predictedOutputDiv.textContent = '';
        predictedOutputDiv.style.display = 'none';
    }
}

// Add the latest prediction to the history list
function addToHistory(prediction, source = 'live') {
    const match = prediction.match(/Model Predicted : (.+)/);
    const signWord = match ? match[1] : prediction;

    const recordsList = document.getElementById('recordsList');
    const newRecord = document.createElement('div');
    newRecord.className = 'record-item';
    newRecord.innerHTML = `
                <div>
                    <div class="record-prediction">${signWord}</div>
                    <div class="record-time">${new Date().toLocaleString()}</div>
                </div>
                <div>
                    <span style="color: #6c757d;">${source === 'live' ? 'Live' : 'Upload'}</span>
                </div>
            `;

    // Always insert new records at the top
    if (recordsList.firstChild) {
        recordsList.insertBefore(newRecord, recordsList.firstChild);
    } else {
        recordsList.appendChild(newRecord);
    }

    // Keep only last 10 history items
    const records = recordsList.querySelectorAll('.record-item');
    if (records.length > 10) {
        recordsList.removeChild(records[records.length - 1]);
    }
}








// Toggle navigation menu
document.getElementById('menuToggle').addEventListener('click', function () {
    document.getElementById('navMenu').classList.toggle('active');
});

// Toggle dark/light theme
function toggleTheme() {
    const html = document.documentElement;
    const currentTheme = html.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    html.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    
    // Update theme icon if it exists
    const icon = document.getElementById("theme-icon");
    if (icon) {
        if (newTheme === 'dark') {
            icon.className = "fas fa-sun";
        } else {
            icon.className = "fas fa-moon";
        }
    }
}

// Initialize theme on page load
(function () {
    const savedTheme = localStorage.getItem('theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
    
    // Set initial icon state
    const icon = document.getElementById("theme-icon");
    if (icon) {
        if (savedTheme === 'dark') {
            icon.className = "fas fa-sun";
        } else {
            icon.className = "fas fa-moon";
        }
    }
})();

// Toggle dark/light theme with icon change
document.addEventListener('DOMContentLoaded', function() {
    const themeToggleBtn = document.getElementById('themeToggle');
    if (themeToggleBtn) {
        themeToggleBtn.addEventListener('click', function () {
            toggleTheme();
        });
    }
});





// Update the frame counter display
function updateFrameCounter() {
    const counter = document.getElementById('frameCounter');
    if (isPredicting) {
        counter.textContent = `Frames Captured: ${capturedFrames.length} / 60 (3 seconds at 20fps)`;
    } else if (isRecording) {
        counter.textContent = 'Camera ready - Click Start Prediction to begin';
    } else if (currentMode === 'upload') {
        counter.textContent = 'No video selected';
    } else {
        counter.textContent = '';
    }
}

// Update the statistics (total predictions made)
function updateStats() {
    document.getElementById('totalPredictions').textContent = totalPredictions;
}

// FPS Counter logic
function startFPSCounter() {
    let lastTime = performance.now();
    let frameCounter = 0;

    fpsInterval = setInterval(() => {
        const now = performance.now();
        const delta = now - lastTime;
        frameCounter++;

        if (delta >= 1000) {
            const fps = Math.round((frameCounter * 1000) / delta);
            document.getElementById('fps').textContent = fps;
            frameCounter = 0;
            lastTime = now;
        }
    }, 100);
}

function stopFPSCounter() {
    if (fpsInterval) {
        clearInterval(fpsInterval);
        document.getElementById('fps').textContent = '0';
    }
}

// Show/hide loading spinner
function showLoading() {
    document.getElementById('loading').classList.add('show');
}

function hideLoading() {
    document.getElementById('loading').classList.remove('show');
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.classList.add('show');
}

function hideError() {
    document.getElementById('errorMessage').classList.remove('show');
}

// Mode switching functionality
function switchMode(mode) {
    currentMode = mode;
    
    // Update button states
    document.getElementById('liveModeBtn').classList.toggle('active', mode === 'live');
    document.getElementById('uploadModeBtn').classList.toggle('active', mode === 'upload');
    
    // Show/hide appropriate controls
    document.getElementById('liveControls').style.display = mode === 'live' ? 'block' : 'none';
    document.getElementById('uploadControls').style.display = mode === 'upload' ? 'block' : 'none';    // Show/hide canvas wrapper based on mode
    const canvasWrapper = document.getElementById('canvasWrapper');
    const videoContainer = document.querySelector('.video-container');
    
    if (canvasWrapper) {
        canvasWrapper.style.display = mode === 'live' ? 'block' : 'none';
    }
    
    // Add/remove upload mode class for styling
    if (videoContainer) {
        if (mode === 'upload') {
            videoContainer.classList.add('upload-mode');
        } else {
            videoContainer.classList.remove('upload-mode');
        }
    }
    
    // Update video title
    const videoTitle = document.getElementById('videoTitle');
    if (mode === 'live') {
        videoTitle.innerHTML = '<i class="fas fa-camera"></i> Live Camera Feed';
    } else {
        videoTitle.innerHTML = '<i class="fas fa-file-video"></i> Uploaded Video';
    }
      // Reset states when switching modes
    if (mode === 'live') {
        clearUploadedVideo();
        document.getElementById('videoElement').style.display = 'block';
        document.getElementById('uploadedVideo').style.display = 'none';
    } else {
        stopCamera();
        document.getElementById('videoElement').style.display = 'none';
        // Only show uploaded video if there's actually a video file
        if (uploadedVideoFile) {
            document.getElementById('uploadedVideo').style.display = 'block';
        } else {
            document.getElementById('uploadedVideo').style.display = 'none';
        }
    }
    
    // Reset prediction section
    document.getElementById('predictionSection').classList.remove('show');
    document.getElementById('signWord').textContent = '-';
    document.getElementById('confidence').textContent = 'Waiting for prediction...';
    hidePredictedOutput();
}

// Clear uploaded video (used when switching to live mode)
function clearUploadedVideo() {
    uploadedVideoFile = null;
    const videoFileInput = document.getElementById('videoFileInput');
    if (videoFileInput) videoFileInput.value = '';
    const uploadInfo = document.getElementById('uploadInfo');
    if (uploadInfo) uploadInfo.style.display = 'none';
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.style.display = 'none';
    }
    const uploadBtn = document.querySelector('.upload-btn');
    if (uploadBtn) uploadBtn.style.display = 'inline-flex';
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    if (cancelUploadBtn) cancelUploadBtn.style.display = 'none';
    const uploadedVideo = document.getElementById('uploadedVideo');
    if (uploadedVideo && uploadedVideo.src) {
        URL.revokeObjectURL(uploadedVideo.src);
        uploadedVideo.src = '';
    }
    if (uploadedVideo) uploadedVideo.style.display = 'none';
    const frameCounter = document.getElementById('frameCounter');
    if (frameCounter) frameCounter.textContent = 'No video selected';
    const uploadStatus = document.getElementById('uploadStatus');
    if (uploadStatus) uploadStatus.style.display = 'none';
    const predictionSection = document.getElementById('predictionSection');
    if (predictionSection) predictionSection.classList.remove('show');
    const signWord = document.getElementById('signWord');
    if (signWord) signWord.textContent = '-';
    const confidence = document.getElementById('confidence');
    if (confidence) confidence.textContent = 'Waiting for prediction...';
    hidePredictedOutput();
}

// Stop camera (utility function for mode switching)
function stopCamera() {
    if (camera) {
        camera.stop();
        camera = null;
    }
    if (videoElement && videoElement.srcObject) {
        videoElement.srcObject.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
    }
    if (videoElement) {
        videoElement.pause();
        videoElement.removeAttribute('src');
        videoElement.load();
    }
    isRecording = false;
    isPredicting = false;
    frameCount = 0;
    capturedFrames = [];
    // Reset UI controls
    const startCameraBtn = document.getElementById('startCameraBtn');
    if (startCameraBtn) {
        startCameraBtn.disabled = false;
        startCameraBtn.style.display = 'inline-flex';
    }
    document.getElementById('startPredictionBtn').disabled = true;
    document.getElementById('startPredictionBtn').style.display = 'none';
    document.getElementById('stopPredictionBtn').disabled = true;
    document.getElementById('stopPredictionBtn').style.display = 'none';
    document.getElementById('stopCameraBtn').disabled = true;
    document.getElementById('stopCameraBtn').style.display = 'none';
    document.getElementById('predictionStatus').style.display = 'none';
    // Clear the canvas so last frame is not shown
    if (canvasCtx && outputCanvas) {
        canvasCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    }
    updateFrameCounter();
    stopFPSCounter();
}

// Handle video file upload
function handleVideoUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    if (!file.type.startsWith('video/')) {
        alert('Please select a valid video file.');
        return;
    }
    
    // Validate file size (limit to 50MB)
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        alert('File size is too large. Please select a video under 50MB.');
        return;
    }
    
    uploadedVideoFile = file;
    
    // Enable analyze button
    document.getElementById('analyzeBtn').disabled = false;
      // Create video URL and set it to the video element
    const videoURL = URL.createObjectURL(file);
    const uploadedVideo = document.getElementById('uploadedVideo');
    uploadedVideo.src = videoURL;
    uploadedVideo.style.display = 'block';  // Show the video now that it's loaded
    
    // Hide upload button, show analyze button
    document.querySelector('.upload-btn').style.display = 'none';
    document.getElementById('analyzeBtn').style.display = 'inline-flex';
    
    // Update frame counter for upload mode
    document.getElementById('frameCounter').textContent = `Video loaded: ${file.name}`;
    
    // Show Cancel button when a file is selected
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    if (cancelUploadBtn) cancelUploadBtn.style.display = 'inline-flex';
}

// Cancel upload and reset UI
function cancelUploadVideo() {
    uploadedVideoFile = null;
    const videoFileInput = document.getElementById('videoFileInput');
    if (videoFileInput) videoFileInput.value = '';
    const uploadInfo = document.getElementById('uploadInfo');
    if (uploadInfo) uploadInfo.style.display = 'none';
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (analyzeBtn) {
        analyzeBtn.disabled = true;
        analyzeBtn.style.display = 'none';
    }
    const uploadBtn = document.querySelector('.upload-btn');
    if (uploadBtn) uploadBtn.style.display = 'inline-flex';
    const cancelUploadBtn = document.getElementById('cancelUploadBtn');
    if (cancelUploadBtn) cancelUploadBtn.style.display = 'none';
    const uploadedVideo = document.getElementById('uploadedVideo');
    if (uploadedVideo && uploadedVideo.src) {
        URL.revokeObjectURL(uploadedVideo.src);
        uploadedVideo.src = '';
    }
    if (uploadedVideo) uploadedVideo.style.display = 'none';
    const frameCounter = document.getElementById('frameCounter');
    if (frameCounter) frameCounter.textContent = 'No video selected';
    const uploadStatus = document.getElementById('uploadStatus');
    if (uploadStatus) uploadStatus.style.display = 'none';
    const predictionSection = document.getElementById('predictionSection');
    if (predictionSection) predictionSection.classList.remove('show');
    const signWord = document.getElementById('signWord');
    if (signWord) signWord.textContent = '-';
    const confidence = document.getElementById('confidence');
    if (confidence) confidence.textContent = 'Waiting for prediction...';
    hidePredictedOutput();
}

// Analyze uploaded video
async function analyzeUploadedVideo() {
    if (!uploadedVideoFile) {
        alert('Please select a video file first.');
        return;
    }
    try {
        document.getElementById('uploadStatus').style.display = 'inline-flex';
        showLoading();
        // Create FormData for file upload
        const formData = new FormData();
        formData.append('video', uploadedVideoFile);
        const response = await fetch('/predict-video', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        if (response.ok) {
            showPrediction(result.prediction);
            showPredictedOutput(result.prediction);
            totalPredictions++;
            updateStats();
            addToHistory(result.prediction, 'upload');
            document.getElementById('predictionSection').classList.add('show');
        } else {
            alert(`Prediction failed: ${result.error || 'Unknown error'}`);
            hidePredictedOutput();
        }
    } catch (error) {
        console.error('Upload error:', error);
        alert('Failed to analyze video. Please try again.');
        hidePredictedOutput();
    } finally {
        document.getElementById('uploadStatus').style.display = 'none';
        hideLoading();
        // Reset uploaded video so a new file can be uploaded without refresh
        uploadedVideoFile = null;
        document.getElementById('videoFileInput').value = '';
        // Hide and clear uploaded video element
        const uploadedVideo = document.getElementById('uploadedVideo');
        if (uploadedVideo) {
            if (uploadedVideo.src) {
                URL.revokeObjectURL(uploadedVideo.src);
                uploadedVideo.src = '';
            }
            uploadedVideo.style.display = 'none';
        }
        document.getElementById('analyzeBtn').disabled = true;
        document.querySelector('.upload-btn').style.display = 'inline-flex';
        document.getElementById('analyzeBtn').style.display = 'none';
        // Hide Cancel button after prediction or clear
        const cancelUploadBtn = document.getElementById('cancelUploadBtn');
        if (cancelUploadBtn) cancelUploadBtn.style.display = 'none';
    }
}

// Format file size helper function
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Initialize on page load
window.addEventListener('load', () => {
    initializeMediaPipe();
});

// On page load, ensure only upload button is visible
window.addEventListener('DOMContentLoaded', function() {
    document.querySelector('.upload-btn').style.display = 'inline-flex';
    document.getElementById('analyzeBtn').style.display = 'none';
    document.getElementById('cancelUploadBtn').addEventListener('click', cancelUploadVideo);
});
