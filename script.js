let model;
let datasetEmbeddings = [];
const posterMapping = {
    "406": "Box 3",
    "402": "Box 3",
    "460": "Box 3"
};

// Load MobileNet model
async function loadModel() {
    model = await mobilenet.load();
    console.log("Model loaded");
}

// Capture video frame safely
async function captureImage() {
    const video = document.getElementById("video");
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        alert("Video not ready yet. Please wait a moment.");
        return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas;
}

// Compute embedding from canvas for MobileNet
async function getEmbedding(canvas) {
    const tensorImg = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
    const embedding = model.infer(tensorImg, true).flatten();
    return embedding;
}

// Find closest poster using embeddings
function findClosest(embedding) {
    let minDist = Infinity;
    let bestPoster = null;

    for (const item of datasetEmbeddings) {
        const dist = tf.norm(tf.sub(embedding, tf.tensor(item.embedding))).dataSync()[0];
        if (dist < minDist) {
            minDist = dist;
            bestPoster = item.poster;
        }
    }
    return bestPoster;
}

// Load embeddings JSON
async function loadDataset() {
    try {
        const response = await fetch("poster_embeddings.json");
        if (!response.ok) throw new Error("Could not load JSON");
        const data = await response.json();
        datasetEmbeddings = data;
        console.log("Dataset embeddings loaded");
    } catch (err) {
        console.error("Error loading dataset embeddings:", err);
        alert("Failed to load poster embeddings. Make sure poster_embeddings.json is accessible.");
    }
}

// Scan poster
async function scanPoster() {
    const canvas = await captureImage();
    if (!canvas) return;

    const embedding = await getEmbedding(canvas);
    const posterNumber = findClosest(embedding);

    if (posterNumber) {
        document.getElementById("result").innerText = 
            `Poster Number: ${posterNumber}\nBox: ${posterMapping[posterNumber]}`;
    } else {
        document.getElementById("result").innerText = "No matching poster found.";
    }
}

// Initialize app
async function init() {
    await loadModel();
    await loadDataset();

    const video = document.getElementById("video");
    
    // Access rear camera for iOS and mobile
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then(stream => {
            video.srcObject = stream;
            video.setAttribute("playsinline", true); // important for iOS
            video.play();
        })
        .catch(err => {
            alert("Camera access denied or not supported: " + err);
            console.error(err);
        });

    document.getElementById("scanBtn").addEventListener("click", scanPoster);
}

// Start everything
init();
