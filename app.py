import os
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the trained CNN model
model = tf.keras.models.load_model('wheat_disease_binary_cnn_model.h5')

# Define class labels for binary classification
class_labels = ['Fusarium Head Blight', 'Healthy Wheat']

# Initialize Flask app
app = Flask(__name__)

# Set up the upload folder for images
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    
    # Save uploaded image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    
    # Preprocess the image
    image = load_img(filepath, target_size=(150, 150))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    # Make prediction
    prediction = model.predict(image)
    confidence_score = float(prediction[0][0])  # Get the confidence score (between 0 and 1)
    
    # Determine the predicted class and adjust confidence score accordingly
    if confidence_score > 0.5:
        predicted_label = 'Healthy Wheat'
        confidence = confidence_score * 100  # Convert to percentage
    else:
        predicted_label = 'Fusarium Head Blight'
        confidence = (1 - confidence_score) * 100  # Convert to percentage

    # Render the result page with the prediction and confidence score
    return render_template('result.html', prediction=predicted_label, confidence=confidence, image_file=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
