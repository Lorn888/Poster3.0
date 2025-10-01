let model;
let datasetEmbeddings = [];
const posterMapping = {
    "406": "Box 3",
    "402": "Box 3",
    "460": "Box 3"
};

// Logging function
function log(message) {
    console.log(message);
    const logDiv = document.getElementById("log");
    logDiv.innerText += message + "\n";
}

// Load MobileNet
async function loadModel() {
    log("Loading MobileNet model...");
    model = await mobilenet.load();
    log("MobileNet model loaded");
}

// Load poster embeddings JSON
async function loadDataset() {
    log("Loading poster embeddings JSON...");
    try {
        const response = await fetch("poster_embeddings.json");
        if (!response.ok) throw new Error("Could not load JSON");
        const data = await response.json();
        datasetEmbeddings = data;
        log(`Loaded ${datasetEmbeddings.length} poster embeddings`);
    } catch (err) {
        log("Error loading dataset embeddings: " + err);
        alert("Failed to load poster embeddings. Check log.");
    }
}

// Capture video frame
async function captureImage() {
    log("Capturing image from camera...");
    const video = document.getElementById("video");
    if (video.videoWidth === 0 || video.videoHeight === 0) {
        log("Video not ready yet");
        alert("Video not ready yet. Please wait a moment.");
        return null;
    }
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    log("Image captured from video");
    return canvas;
}

// Compute embedding
async function getEmbedding(canvas) {
    log("Computing embedding...");
    const tensorImg = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
    const embedding = model.infer(tensorImg, true).flatten();
    log("Embedding computed");
    return embedding;
}

// Find closest poster
function findClosest(embedding) {
    log("Finding closest poster...");
    let minDist = Infinity;
    let bestPoster = null;

    for (const item of datasetEmbeddings) {
        const dist = tf.norm(tf.sub(embedding, tf.tensor(item.embedding))).dataSync()[0];
        if (dist < minDist) {
            minDist = dist;
            bestPoster = item.poster;
        }
    }
    log("Closest poster found: " + bestPoster);
    return bestPoster;
}

// Scan poster
async function scanPoster() {
    log("\n--- Scan started ---");
    const canvas = await captureImage();
    if (!canvas) {
        log("Scan aborted: no image captured");
        return;
    }

    const embedding = await getEmbedding(canvas);
    const posterNumber = findClosest(embedding);

    if (posterNumber) {
        log(`Poster Number: ${posterNumber}, Box: ${posterMapping[posterNumber]}`);
        document.getElementById("result").innerText = 
            `Poster Number: ${posterNumber}\nBox: ${posterMapping[posterNumber]}`;
    } else {
        log("No matching poster found");
        document.getElementById("result").innerText = "No matching poster found.";
    }
}

// Initialize app
async function init() {
    await loadModel();
    await loadDataset();

    const video = document.getElementById("video");

    // Access rear camera on iPhone/iPad
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then(stream => {
            video.srcObject = stream;
            video.setAttribute("playsinline", true); // important for iOS
            video.play();
            log("Camera stream started");
        })
        .catch(err => {
            log("Camera access denied or not supported: " + err);
            alert("Camera access denied or not supported");
        });

    document.getElementById("scanBtn").addEventListener("click", scanPoster);
}

// Start app
init();
