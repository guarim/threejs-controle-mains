// Import the HandLandmarker class and other necessary modules
import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.js";

const video = document.getElementById("webcam");
const canvasElement = document.getElementById("output_canvas");
const canvasCtx = canvasElement.getContext("2d");
const loadingMessage = document.getElementById("loading-message");

let handLandmarker = undefined;
let runningMode = "VIDEO"; // Can be "IMAGE" or "VIDEO"
let webcamRunning = false;

// Before we can use HandLandmarker class we must task the createFrom... options.
async function createHandLandmarker() {
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
        );
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
            baseOptions: {
                modelAssetPath: `./models/hand_landmarker.task`, // Path to your downloaded model
                delegate: "GPU"
            },
            runningMode: runningMode,
            numHands: 2, // Max number of hands to detect
            minHandDetectionConfidence: 0.5,
            minHandTrackingConfidence: 0.5
        });
        loadingMessage.innerText = "Model loaded. Accessing webcam...";
        enableCam();
    } catch (error) {
        console.error("Error creating HandLandmarker:", error);
        loadingMessage.innerText = "Error loading model. Check console.";
    }
}
createHandLandmarker();

// Enable the live webcam view and start detection.
function enableCam() {
    if (!handLandmarker) {
        console.log("Wait! objectDetector not loaded yet.");
        return;
    }

    // Get user media constraints
    const constraints = {
        video: { width: { ideal: 1280 }, height: { ideal: 720 } } // Request HD but browser may adjust
    };

    // Activate the webcam stream.
    navigator.mediaDevices.getUserMedia(constraints)
        .then(function (stream) {
            video.srcObject = stream;
            video.addEventListener("loadeddata", () => {
                webcamRunning = true;
                loadingMessage.style.display = "none"; // Hide loading message
                predictWebcam();
            });
        })
        .catch(function (err) {
            console.error("Error accessing webcam: ", err);
            loadingMessage.innerText = "Error accessing webcam. Please grant permissions and refresh.";
        });
}

let lastVideoTime = -1;
async function predictWebcam() {
    // Set canvas dimensions to match video
    canvasElement.width = video.videoWidth;
    canvasElement.height = video.videoHeight;

    // Start detecting the stream.
    if (runningMode === "IMAGE") {
        runningMode = "VIDEO";
        await handLandmarker.setOptions({ runningMode: "VIDEO" });
    }
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;
        const results = handLandmarker.detectForVideo(video, startTimeMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);

        if (results.landmarks && results.landmarks.length > 0) {
            for (const landmarks of results.landmarks) {
                drawLandmarks(landmarks);
            }
        }
        canvasCtx.restore();
    }

    // Call this function again to keep predicting when the browser is ready.
    if (webcamRunning) {
        window.requestAnimationFrame(predictWebcam);
    }
}

function drawLandmarks(landmarks) {
    canvasCtx.fillStyle = "#FF0000"; // Red color for landmarks
    canvasCtx.strokeStyle = "#00FF00"; // Green color for connections (optional)
    canvasCtx.lineWidth = 2;

    for (const landmark of landmarks) {
        const x = landmark.x * canvasElement.width;
        const y = landmark.y * canvasElement.height;
        // z is also available (landmark.z) but not used for simple 2D drawing

        canvasCtx.beginPath();
        canvasCtx.arc(x, y, 5, 0, 2 * Math.PI); // Draw a small circle for each landmark
        canvasCtx.fill();
    }

    // Optional: Draw connections (example for a few common connections)
    // You can find the HandLandmarker.HAND_CONNECTIONS array in MediaPipe documentation
    // to draw all standard connections if needed. For simplicity, we'll skip comprehensive connections here.
    /*
    const connections = [
        [0, 1], [1, 2], [2, 3], [3, 4], // Thumb
        [0, 5], [5, 6], [6, 7], [7, 8], // Index finger
        // ... and so on for other fingers and palm
    ];
    for (const connection of connections) {
        const start = landmarks[connection[0]];
        const end = landmarks[connection[1]];
        canvasCtx.beginPath();
        canvasCtx.moveTo(start.x * canvasElement.width, start.y * canvasElement.height);
        canvasCtx.lineTo(end.x * canvasElement.width, end.y * canvasElement.height);
        canvasCtx.stroke();
    }
    */
}

// Add event listener for window resize (optional, but good for responsiveness)
window.addEventListener('resize', () => {
    if (webcamRunning) {
        // Recalculate canvas size on resize, could also re-init aspects of prediction if needed
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }
});