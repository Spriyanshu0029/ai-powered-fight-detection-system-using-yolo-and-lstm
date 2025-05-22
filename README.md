# ai-powered-fight-detection-system-using-yolo-and-lstm
<!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fight Detection System</title>
            <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/coco-ssd"></script>
            <script src="https://cdn.jsdelivr.net/npm/@tensorflow-models/posenet"></script>
            <style>
                body {
                    background: linear-gradient(135deg, rgba(26, 26, 46, 0.9) 0%, rgba(22, 33, 62, 0.9) 100%),
                                url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                    background-repeat: no-repeat;
                }
                
                .detection-box {
                    position: absolute;
                    border: 2px solid #ffd700;
                    color: #ffd700;
                    font-weight: bold;
                    box-shadow: 0 0 10px rgba(255, 215, 0, 0.5);
                }
                
                .warning-banner {
                    background: linear-gradient(90deg, #ffa500 0%, #ff8c00 100%);
                    animation: pulse 2s infinite;
                }
                
                .alert-banner {
                    animation: pulse 1s infinite;
                    background: linear-gradient(90deg, #ff4d4d 0%, #ff0000 100%);
                }
                
                @keyframes pulse {
                    0% { opacity: 0.8; }
                    50% { opacity: 1; }
                    100% { opacity: 0.8; }
                }
                
                .login-container {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
                    position: relative;
                    overflow: hidden;
                }

                .login-container::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: url('https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=2070&q=80');
                    background-size: cover;
                    background-position: center;
                    opacity: 0.1;
                    z-index: -1;
                }
                
                .header-gradient {
                    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                }
                
                .detection-screen {
                    background: rgba(26, 26, 46, 0.95);
                }
                
                .sidebar {
                    background: rgba(255, 255, 255, 0.95);
                    backdrop-filter: blur(10px);
                }
                
                .footer {
                    background: rgba(22, 33, 62, 0.95);
                    backdrop-filter: blur(10px);
                }

                .live-indicator {
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    background: rgba(0, 0, 0, 0.7);
                    padding: 6px 12px;
                    border-radius: 20px;
                    color: white;
                    font-size: 14px;
                    z-index: 10;
                }

                .live-dot {
                    width: 8px;
                    height: 8px;
                    background-color: #ff4444;
                    border-radius: 50%;
                    animation: live-pulse 1.5s infinite;
                }

                @keyframes live-pulse {
                    0% { transform: scale(1); opacity: 1; }
                    50% { transform: scale(1.5); opacity: 0.5; }
                    100% { transform: scale(1); opacity: 1; }
                }

                .camera-frame {
                    position: relative;
                    border: 2px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    overflow: hidden;
                    box-shadow: 0 0 20px rgba(0, 0, 0, 0.3);
                }

                .camera-frame::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    border: 2px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    pointer-events: none;
                    animation: frame-pulse 2s infinite;
                }

                @keyframes frame-pulse {
                    0% { border-color: rgba(255, 255, 255, 0.1); }
                    50% { border-color: rgba(255, 255, 255, 0.3); }
                    100% { border-color: rgba(255, 255, 255, 0.1); }
                }

                .detection-stats {
                    position: absolute;
                    bottom: 10px;
                    left: 10px;
                    background: rgba(0, 0, 0, 0.7);
                    padding: 8px 12px;
                    border-radius: 8px;
                    color: white;
                    font-size: 14px;
                    z-index: 10;
                }
            </style>
        </head>
        <body class="min-h-screen">
            <div id="login-screen" class="flex items-center justify-center min-h-screen p-4">
                <div class="w-full max-w-md p-8 space-y-8 login-container rounded-xl">
                    <div class="text-center">
                        <div class="mx-auto h-16 w-16 mb-4 text-blue-600">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                            </svg>
                        </div>
                        <h2 class="text-3xl font-bold text-gray-900">Fight Detection System</h2>
                        <p class="mt-2 text-sm text-gray-600">Please sign in to continue</p>
                    </div>
                    
                    <div id="error-message" class="p-3 text-sm text-red-500 bg-red-100 rounded hidden"></div>
                    
                    <div class="mt-8 space-y-6">
                        <div class="rounded-md shadow-sm space-y-4">
                            <div class="relative">
                                <label for="username" class="sr-only">Username</label>
                                <div class="absolute left-3 top-3 h-5 w-5 text-gray-400">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                                    </svg>
                                </div>
                                <input
                                    id="username"
                                    name="username"
                                    type="text"
                                    class="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Username"
                                />
                            </div>
                            
                            <div class="relative">
                                <label for="password" class="sr-only">Password</label>
                                <div class="absolute left-3 top-3 h-5 w-5 text-gray-400">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                                    </svg>
                                </div>
                                <input
                                    id="password"
                                    name="password"
                                    type="password"
                                    class="pl-10 w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                    placeholder="Password"
                                />
                                <button
                                    id="toggle-password"
                                    type="button"
                                    class="absolute inset-y-0 right-0 flex items-center pr-3"
                                >
                                    <svg id="eye-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" class="h-5 w-5 text-gray-400">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                    </svg>
                                    <svg id="eye-off-icon" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" class="h-5 w-5 text-gray-400 hidden">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                                    </svg>
                                </button>
                            </div>
                        </div>
                        
                        <div>
                            <button
                                id="login-button"
                                class="group relative w-full flex justify-center py-2 px-4 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                            >
                                Sign in
                            </button>
                        </div>
                        
                        <div class="text-sm text-center">
                            <p class="text-gray-500">Demo credentials: priyanshu / 12345678</p>
                        </div>
                    </div>
                </div>
            </div>

            <div id="detection-screen" class="hidden flex flex-col h-screen">
                <!-- Header -->
                <header class="header-gradient text-white p-4 shadow-lg">
                    <div class="flex justify-between items-center">
                        <div class="flex items-center space-x-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                            </svg>
                            <h1 class="text-lg font-bold">Fight Detection System</h1>
                        </div>
                        <button id="logout-button" class="px-4 py-2 bg-blue-700 rounded-lg hover:bg-blue-800 transition-colors duration-200 flex items-center space-x-2">
                            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
                            </svg>
                            <span>Logout</span>
                        </button>
                    </div>
                </header>
                
                <!-- Warning Banner -->
                <div id="warning-banner" class="hidden bg-orange-500 text-white p-2 text-center font-bold warning-banner">
                    ⚠ POTENTIAL THREAT DETECTED! PROBABILITY: <span id="warning-probability">0</span>%
                </div>
                
                <!-- Alert Banner -->
                <div id="alert-banner" class="hidden bg-red-600 text-white p-2 text-center font-bold alert-banner">
                    ⚠ FIGHT DETECTED! PROBABILITY: <span id="fight-probability">0</span>%
                </div>
                
                <!-- Main content -->
                <main class="flex flex-col md:flex-row flex-1 overflow-hidden">
                    <!-- Video feed -->
                    <div class="flex-1 p-4 flex flex-col">
                        <div class="relative bg-black rounded-lg overflow-hidden flex-1 flex items-center justify-center camera-frame">
                            <div id="camera-off-message" class="text-center text-white">
                                <div class="mx-auto h-12 w-12 mb-4">
                                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                                    </svg>
                                </div>
                                <p>Camera is turned off</p>
                            </div>
                            <div id="video-container" class="relative w-full h-full hidden">
                                <div class="live-indicator">
                                    <div class="live-dot"></div>
                                    <span>LIVE</span>
                                </div>
                                <div class="detection-stats">
                                    <div>Persons Detected: <span id="persons-count">0</span></div>
                                    <div>FPS: <span id="fps-counter">0</span></div>
                                </div>
                                <video id="video" class="h-full w-full object-cover" autoplay muted playsinline></video>
                                <canvas id="detection-canvas" class="absolute top-0 left-0 w-full h-full"></canvas>
                            </div>
                        </div>
                        
                        <div class="mt-4 flex justify-center">
                            <button
                                id="toggle-detection"
                                class="px-4 py-2 rounded-lg bg-green-500 hover:bg-green-600 text-white"
                            >
                                Start Detection
                            </button>
                        </div>
                    </div>
                    
                    <!-- Detection sidebar -->
                    <div class="w-full md:w-64 p-4 bg-white shadow-inner overflow-y-auto">
                        <h2 class="text-lg font-semibold mb-4">Recent Detections</h2>
                        
                        <div id="no-detections" class="text-gray-500 text-sm">No detections yet</div>
                        
                        <div id="detections-list" class="space-y-3">
                            <!-- Detection items will be added here -->
                        </div>
                    </div>
                </main>
                
                <!-- Footer -->
                <footer class="footer p-3 text-center text-sm text-gray-400">
                    <div class="flex items-center justify-center space-x-2">
                        <svg xmlns="http://www.w3.org/2000/svg" class="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                        </svg>
                        <span>Fight Detection System v1.0</span>
                    </div>
                </footer>
            </div>

            <!-- Alert sound -->
            <audio id="alert-sound" preload="auto">
                <source src="raid.mp3" type="audio/mpeg">
            </audio>

        <script>
        let cocoModel = null;
        let poseNetModel = null;
        let isAlertPlaying = false;
        let alertSoundPromise = null;
            
        async function loadModels() {
            cocoModel = await cocoSsd.load();
            poseNetModel = await posenet.load();
        }

        // Function to play alert sound safely
        async function playAlertSound() {
            const alertSound = document.getElementById('alert-sound');
            
            try {
                // If there's an ongoing play promise, wait for it to complete
                if (alertSoundPromise) {
                    await alertSoundPromise;
                }
                
                // Reset the audio to the beginning
                alertSound.currentTime = 0;
                
                // Create a new play promise
                alertSoundPromise = alertSound.play();
                await alertSoundPromise;
                
                // Set up an event listener for when the sound ends
                alertSound.onended = () => {
                    isAlertPlaying = false;
                    alertSoundPromise = null;
                };
                
                isAlertPlaying = true;
            } catch (error) {
                console.error('Error playing alert sound:', error);
                isAlertPlaying = false;
                alertSoundPromise = null;
            }
        }

        loadModels();
            // Login functionality
            document.getElementById('login-button').addEventListener('click', function() {
                const username = document.getElementById('username').value;
                const password = document.getElementById('password').value;
                const errorMessage = document.getElementById('error-message');
                
                if (!username || !password) {
                    errorMessage.textContent = 'Please enter both username and password';
                    errorMessage.classList.remove('hidden');
                    return;
                }
                
                // Simple validation - in a real app, this would connect to a backend
                if (username === 'priyanshu' && password === '12345678') {
                    document.getElementById('login-screen').classList.add('hidden');
                    document.getElementById('detection-screen').classList.remove('hidden');
                    errorMessage.classList.add('hidden');
                } else {
                    errorMessage.textContent = 'Invalid credentials. Try priyanshu/12345678';
                    errorMessage.classList.remove('hidden');
                }
            });
            
            // Password visibility toggle
            document.getElementById('toggle-password').addEventListener('click', function() {
                const passwordInput = document.getElementById('password');
                const eyeIcon = document.getElementById('eye-icon');
                const eyeOffIcon = document.getElementById('eye-off-icon');
                
                if (passwordInput.type === 'password') {
                    passwordInput.type = 'text';
                    eyeIcon.classList.add('hidden');
                    eyeOffIcon.classList.remove('hidden');
                } else {
                    passwordInput.type = 'password';
                    eyeIcon.classList.remove('hidden');
                    eyeOffIcon.classList.add('hidden');
                }
            });
            
            // Logout functionality
            document.getElementById('logout-button').addEventListener('click', function() {
                document.getElementById('login-screen').classList.remove('hidden');
                document.getElementById('detection-screen').classList.add('hidden');
                stopDetection();
            });
            
            // Get references to elements
            const video = document.getElementById('video');
            const videoContainer = document.getElementById('video-container');
            const cameraOffMessage = document.getElementById('camera-off-message');
            const toggleDetectionButton = document.getElementById('toggle-detection');
            const detectionCanvas = document.getElementById('detection-canvas');
            const detectionsListElement = document.getElementById('detections-list');
            const noDetectionsElement = document.getElementById('no-detections');
            const warningBanner = document.getElementById('warning-banner');
            const warningProbability = document.getElementById('warning-probability');
            const alertBanner = document.getElementById('alert-banner');
            const fightProbabilityElement = document.getElementById('fight-probability');
            const alertSound = document.getElementById('alert-sound');
            
            // Global variables
            let isDetecting = false;
            let model = null;
            let stream = null;
            let animationId = null;
            let lastAlertTime = 0;
            let detections = [];
            let frameCount = 0;
            let lastFpsUpdate = 0;
            let fps = 0;
            
            // Update FPS counter
            function updateFps() {
                frameCount++;
                const now = performance.now();
                if (now - lastFpsUpdate >= 1000) {
                    fps = Math.round((frameCount * 1000) / (now - lastFpsUpdate));
                    document.getElementById('fps-counter').textContent = fps;
                    frameCount = 0;
                    lastFpsUpdate = now;
                }
            }
            
            // Start camera and detection
            async function startDetection() {
                try {
                    // Load COCO-SSD model (as a replacement for YOLO)
                    if (!model) {
                        model = await cocoSsd.load();
                    }
                    
                    // Access camera
                    stream = await navigator.mediaDevices.getUserMedia({
                        video: { 
                            facingMode: 'environment',
                            width: { ideal: 640 },
                            height: { ideal: 480 }
                        }
                    });
                    
                    // Set up video stream
                    video.srcObject = stream;
                    await video.play();
                    
                    // Set canvas size
                    detectionCanvas.width = video.videoWidth;
                    detectionCanvas.height = video.videoHeight;
                    
                    // Update UI
                    videoContainer.classList.remove('hidden');
                    cameraOffMessage.classList.add('hidden');
                    toggleDetectionButton.textContent = 'Stop Detection';
                    toggleDetectionButton.classList.remove('bg-green-500', 'hover:bg-green-600');
                    toggleDetectionButton.classList.add('bg-red-500', 'hover:bg-red-600');
                    
                    isDetecting = true;
                    
                    // Start detection loop
                    detectFrame();
                } catch (error) {
                    console.error('Error starting detection:', error);
                    alert('Could not access camera. Please ensure camera permissions are granted.');
                }
            }
            
            // Stop camera and detection
            function stopDetection() {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                    stream = null;
                }
                
                if (animationId) {
                    cancelAnimationFrame(animationId);
                    animationId = null;
                }
                
                // Update UI
                videoContainer.classList.add('hidden');
                cameraOffMessage.classList.remove('hidden');
                toggleDetectionButton.textContent = 'Start Detection';
                toggleDetectionButton.classList.remove('bg-red-500', 'hover:bg-red-600');
                toggleDetectionButton.classList.add('bg-green-500', 'hover:bg-green-600');
                
                // Hide alert banners
                warningBanner.classList.add('hidden');
                warningProbability.textContent = '0';
                alertBanner.classList.add('hidden');
                
                isDetecting = false;
            }
            
            // Toggle detection
            toggleDetectionButton.addEventListener('click', function() {
                if (isDetecting) {
                    stopDetection();
                } else {
                    startDetection();
                }
            });
            
            // Calculate center of bounding box
            function getCenter(box) {
                return [
                    (box[0] + box[2]) / 2,
                    (box[1] + box[3]) / 2
                ];
            }
            
            // Calculate distance between two points
            function calculateDistance(point1, point2) {
                return Math.sqrt(
                    Math.pow(point1[0] - point2[0], 2) + 
                    Math.pow(point1[1] - point2[1], 2)
                );
            }
            
            // Calculate fight probability with enhanced gesture detection
            function calculateFightProbability(personBoxes, predictions) {
                let fightScore = 0;
                let numPairs = 0;
                
                // Check for weapons first
                const weaponClasses = ['knife', 'scissors', 'gun', 'pistol', 'rifle', 'bow', 'arrow', 'brass knuckles', 'punch iron'];
                const weapons = predictions.filter(p => 
                    weaponClasses.includes(p.class.toLowerCase()) && 
                    p.score > 0.4
                );
                
                if (weapons.length > 0) {
                    return 100; // Immediate high probability if weapons detected
                }

                // Track person movements and gestures
                for (let i = 0; i < personBoxes.length; i++) {
                    for (let j = i + 1; j < personBoxes.length; j++) {
                        const box1 = personBoxes[i];
                        const box2 = personBoxes[j];
                        
                        const center1 = getCenter(box1);
                        const center2 = getCenter(box2);
                        const dist = calculateDistance(center1, center2);
                        
                        // Only analyze if people are close enough to interact
                        if (dist < 200) {
                            // Calculate relative positions and movements
                            const overlap = calculateOverlap(box1, box2);
                            const heightRatio = Math.max(box1[3], box2[3]) / Math.min(box1[3], box2[3]);
                            
                            // Detect aggressive gestures
                            let gestureScore = 0;
                            
                            // 1. Check for raised arms (aggressive posture)
                            const armRaised1 = box1[3] > box1[2] * 1.8; // Taller than wide
                            const armRaised2 = box2[3] > box2[2] * 1.8;
                            
                            if (armRaised1 || armRaised2) {
                                gestureScore += 20;
                            }
                            
                            // 2. Check for close proximity with raised arms
                            if (dist < 100 && (armRaised1 || armRaised2)) {
                                gestureScore += 30;
                            }
                            
                            // 3. Check for rapid movement (stored in previous frame)
                            if (window.previousBoxes) {
                                const prevBox1 = window.previousBoxes[i];
                                const prevBox2 = window.previousBoxes[j];
                                
                                if (prevBox1 && prevBox2) {
                                    const movement1 = calculateDistance(
                                        getCenter(box1),
                                        getCenter(prevBox1)
                                    );
                                    const movement2 = calculateDistance(
                                        getCenter(box2),
                                        getCenter(prevBox2)
                                    );
                                    
                                    // Rapid movement detection
                                    if (movement1 > 20 || movement2 > 20) {
                                        gestureScore += 25;
                                    }
                                }
                            }
                            
                            // 4. Check for physical contact
                            if (overlap > 0.15) {
                                gestureScore += 35;
                            }
                            
                            // 5. Check for aggressive postures
                            if (heightRatio > 1.3) {
                                gestureScore += 15;
                            }
                            
                            // Combine all factors
                            const finalScore = Math.min(gestureScore, 100);
                            
                            // Only count as fight if multiple aggressive indicators are present
                            if (gestureScore > 40) {
                                fightScore += finalScore;
                                numPairs++;
                            }
                        }
                    }
                }
                
                // Store current boxes for next frame's movement detection
                window.previousBoxes = personBoxes.map(box => [...box]);
                
                return numPairs ? Math.round(fightScore / numPairs) : 0;
            }

            // Calculate overlap between two bounding boxes
            function calculateOverlap(box1, box2) {
                const [x1, y1, w1, h1] = box1;
                const [x2, y2, w2, h2] = box2;
                
                const xOverlap = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
                const yOverlap = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
                
                const overlapArea = xOverlap * yOverlap;
                const box1Area = w1 * h1;
                const box2Area = w2 * h2;
                
                return overlapArea / Math.min(box1Area, box2Area);
            }

            // Main detection function
            async function detectFrame() {
                if (!isDetecting || !cocoModel || !poseNetModel) return;

                try {
                    // COCO-SSD: for person/weapon detection
                    const predictions = await cocoModel.detect(video, 20, 0.3);

                    // PoseNet: for gesture/fight detection
                    const poses = await poseNetModel.estimateMultiplePoses(video, {
                        flipHorizontal: false,
                        maxDetections: 4,
                        scoreThreshold: 0.5,
                        nmsRadius: 20
                    });
                    
                    updateFps();
                    
                    // Filter for persons with confidence > 0.5
                    const personBoxes = predictions
                        .filter(p => p.class === 'person' && p.score > 0.5)
                        .map(p => p.bbox);
                    
                    // Update persons count
                    document.getElementById('persons-count').textContent = personBoxes.length;
                    
                    // Only calculate and show probability if two or more people are detected
                    let fightProbability = 0;
                    if (personBoxes.length >= 2) {
                        fightProbability = calculateFightProbability(personBoxes, predictions);
                    }
                    
                    // Default: hide all banners and probability
                    warningBanner.classList.add('hidden');
                    alertBanner.classList.add('hidden');
                    warningProbability.textContent = '';
                    fightProbabilityElement.textContent = '';
                    
                    // Only show banners and probability if fightProbability > 30 and at least 2 people
                  if (personBoxes.length >= 2 && fightProbability > 30) {
    const now = Date.now();

    if (fightProbability > 90) { // High threat - trigger alarm
        if (now - lastAlertTime > 2000 && !isAlertPlaying) {  // Only play alert every 2 seconds
            playAlertSound();
            lastAlertTime = now;
        }

        // Show alert banner
        warningBanner.classList.add('hidden');
        alertBanner.classList.remove('hidden');

        const weaponClasses = ['knife', 'scissors', 'gun', 'pistol', 'rifle', 'bow', 'arrow', 'brass knuckles', 'punch iron'];
        const alertMessage = predictions.some(p => 
            weaponClasses.includes(p.class.toLowerCase())
        ) ? '⚠ WEAPON DETECTED!' : '⚠ FIGHT DETECTED!';
        
        fightProbabilityElement.textContent = alertMessage + ' PROBABILITY: ' + fightProbability + '%';
    } else { // Medium threat
        // ✅ NEW: Play sound when 50% or more
        if (fightProbability >= 50 && now - lastAlertTime > 2000 && !isAlertPlaying) {
            playAlertSound();
            lastAlertTime = now;
        }

        warningBanner.classList.remove('hidden');
        alertBanner.classList.add('hidden');
        warningProbability.textContent = fightProbability;
    }
}

                    // Draw detections with enhanced visualization
                    const ctx = detectionCanvas.getContext('2d');
                    ctx.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
                    
                    // Draw all detections
                    predictions.forEach(prediction => {
                        if (prediction.score > 0.4) {
                            const [x, y, width, height] = prediction.bbox;
                            const weaponClasses = ['knife', 'scissors', 'gun', 'pistol', 'rifle', 'bow', 'arrow', 'brass knuckles', 'punch iron'];
                            // Set color based on detection type and threat level
                            let color = 'yellow';
                            if (prediction.class === 'person') {
                                if (fightProbability > 45) {
                                    color = 'red'; // High threat
                                } else if (fightProbability > 30) {
                                    color = 'orange'; // Medium threat
                                } else if (height > width * 1.8) {
                                    color = 'orange'; // Potential aggressive posture
                                }
                            } else if (weaponClasses.includes(prediction.class.toLowerCase())) {
                                color = 'red';
                            }
                            // Draw box
                            ctx.strokeStyle = color;
                            ctx.lineWidth = 2;
                            ctx.strokeRect(x, y, width, height);
                            // Draw label
                            ctx.fillStyle = color;
                            ctx.font = '16px Arial';
                            ctx.fillText(prediction.class + ' ' + (prediction.score * 100).toFixed(0) + '%', x, y - 5);
                        }
                    });
                    
                    // Set color based on fight probability
                    let fightTextColor = 'lime';
                    if (fightProbability >= 45) {
                        fightTextColor = 'red';
                    } else if (fightProbability >= 30) {
                        fightTextColor = 'orange';
                    }
                    
                    // Only show probability text on canvas if fightProbability > 30 and at least 2 people
                    if (personBoxes.length >= 2 && fightProbability > 30) {
                        ctx.fillStyle = fightTextColor;
                        ctx.font = 'bold 24px Arial';
                        ctx.fillText('⚠ Threat Level: ' + fightProbability + '%', 20, 40);
                    }
                    
                    // Add detection to list if there's a significant probability
                    if (personBoxes.length >= 2 && fightProbability > 30 && (personBoxes.length > 0 || predictions.some(p => {
                        const weaponClasses = ['knife', 'scissors', 'gun', 'pistol', 'rifle', 'bow', 'arrow', 'brass knuckles', 'punch iron'];
                        return weaponClasses.includes(p.class.toLowerCase());
                    }))) {
                        addDetection({
                            id: Date.now(),
                            type: predictions.some(p => {
                                const weaponClasses = ['knife', 'scissors', 'gun', 'pistol', 'rifle', 'bow', 'arrow', 'brass knuckles', 'punch iron'];
                                return weaponClasses.includes(p.class.toLowerCase());
                            }) ? 'Weapon Detected' : (fightProbability > 45 ? 'Fight Detected' : 'Potential Threat'),
                            confidence: fightProbability + '%',
                            timestamp: new Date().toLocaleTimeString(),
                            persons: personBoxes.length
                        });
                    }
                } catch (error) {
                    console.error('Error in detection:', error);
                }
                
                // Continue detection loop
                animationId = requestAnimationFrame(detectFrame);
            }
            
            // Add detection to sidebar
            function addDetection(detection) {
                // Keep only last 5 detections
                detections.unshift(detection);
                if (detections.length > 5) {
                    detections.pop();
                }
                
                // Update UI
                noDetectionsElement.classList.add('hidden');
                updateDetectionsList();
            }
            
            // Update detections list in sidebar
            function updateDetectionsList() {
                detectionsListElement.innerHTML = '';
                
                detections.forEach(detection => {
                    const detectionElement = document.createElement('div');
                    detectionElement.className = 'p-3 border rounded-lg bg-gray-50';
                    detectionElement.innerHTML = `
                        <div class="flex justify-between">
                            <span class="font-medium">
                                ${detection.type} (${detection.persons} persons)
                            </span>
                            <span class="text-sm ${parseInt(detection.confidence) >= 10 ? 'text-red-600' : 'text-green-600'}">
                                ${detection.confidence}
                            </span>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">
                            ${detection.timestamp}
                        </div>
                    `;
                    detectionsListElement.appendChild(detectionElement);
                });
            }
        </script>
    </body>
    </html>
