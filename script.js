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
    const response = await fetch("poster_embeddings.json");
    const data = await response.json();
    datasetEmbeddings = data;
    console.log("Dataset embeddings loaded");
}

// Scan poster
async function scanPoster() {
    const img = await captureImage();
    const embedding = await getEmbedding(tf.browser.fromPixels(img));
    const posterNumber = findClosest(embedding);
    document.getElementById("result").innerText = 
        `Poster Number: ${posterNumber}\nBox: ${posterMapping[posterNumber]}`;
}

// Initialize app
async function init() {
    await loadModel();
    await loadDataset();

    const video = document.getElementById("video");
    navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
        video.srcObject = stream;
    });

    document.getElementById("scanBtn").addEventListener("click", scanPoster);
}

init();