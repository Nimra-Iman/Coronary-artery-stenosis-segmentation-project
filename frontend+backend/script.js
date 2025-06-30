async function uploadImage() {
    const input = document.getElementById('imageInput');
    const file = input.files[0];

    if (!file) {
        alert("Please select an image.");
        return;
    }

    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.prediction) {
            const img = document.getElementById('predictionImage');
            img.src = `data:image/png;base64,${data.prediction}`;
        } else {
            alert("Prediction failed.");
        }
    } catch (error) {
        alert("Error: " + error.message);
    }
}
