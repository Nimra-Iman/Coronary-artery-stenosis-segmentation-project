from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.keras.models import load_model
from keras import backend as K


from PIL import Image
import io
import base64
import cv2



# ****************   CUSTOM LOSS FUNCTION    *******************
# Custom objects for model loading (if any, such as custom loss or layers)
def focal_loss(y_true, y_pred, alpha=0.6, gamma=2):

    y_pred = K.clip(y_pred, 1e-6, 1 - 1e-6)  # Avoid log(0) issue

    # Compute focal loss for each pixel
    loss = -y_true * alpha * K.pow(1 - y_pred, gamma) * K.log(y_pred)

    return K.sum(loss, axis=-1)  # Sum over all pixels

def dice_loss(y_true, y_pred):

    smooth = 1e-6  # Small value to prevent division by zero

    # Flatten only the stenosis class (class 1)
    y_true_f = tf.keras.backend.flatten(y_true[..., 1]) # This converts the 2D mask (height × width) into a 1D
                                                          # array for easier mathematical operations.
    y_pred_f = tf.keras.backend.flatten(y_pred[..., 1])

    # Compute Dice coefficient
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_coeff  # Dice loss = 1 - Dice coefficient

def combined_loss(y_true, y_pred, a=0.6):

    return a * focal_loss(y_true, y_pred) + (1 - a) * dice_loss(y_true, y_pred)



# ***********  KERAS MODEL LOADING   *******************


# Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model

model_path = "C:\\code_fun\\uni ML project\\TECHNICAL PAPER\\61.42F_seUnet.keras"
model = load_model(model_path, custom_objects={'combined_loss': combined_loss})




# Image preprocessing function
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image



# Convert prediction mask to image
def postprocess_prediction(pred):
    # Assume pred shape is (1, 128, 128, num_classes)
    # Get class with highest probability per pixel (e.g., argmax if multi-class, or threshold if binary)

    if pred.shape[-1] == 1:  # binary segmentation
        mask = (pred[0, ..., 0] > 0.5).astype(np.uint8) * 255
    else:  # multi-class segmentation
        mask = np.argmax(pred[0], axis=-1).astype(np.uint8) * 127  # scale for visibility

    # Encode the mask image as PNG
    _, pred_encoded = cv2.imencode(".png", mask)
    base64_result = base64.b64encode(pred_encoded).decode('utf-8')
    return base64_result


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    # Preprocess image
    preprocessed = preprocess_image(image_bytes)

    try:
        prediction = model.predict(preprocessed)
        result_b64 = postprocess_prediction(prediction)
        return jsonify({'prediction': result_b64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)





