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

// Capture image from video
async function captureImage() {
    const video = document.getElementById("video");
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas;
}

// Compute embedding of an image
async function getEmbedding(img) {
    const activation = model.infer(img, true);
    return activation.flatten();
}

// Find closest poster from embeddings
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

// Load JSON embeddings
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
    try {
        const img = await captureImage();
        const embedding = await getEmbedding(tf.browser.fromPixels(img));
        const posterNumber = findClosest(embedding);
        document.getElementById("result").innerText = 
            `Poster Number: ${posterNumber}\nBox: ${posterMapping[posterNumber]}`;
    } catch (err) {
        console.error("Error scanning poster:", err);
    }
}

// Initialize app
async function init() {
    await loadModel();
    await loadDataset();

    const video = document.getElementById("video");
    
    // Use rear camera on iPhone/iPad
    navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } })
        .then(stream => {
            video.srcObject = stream;
            video.play();
        })
        .catch(err => {
            alert("Camera access denied or not supported: " + err);
            console.error(err);
        });

    document.getElementById("scanBtn").addEventListener("click", scanPoster);
}

// Start the app
init();
