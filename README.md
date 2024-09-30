# Wheat Disease Detection

This project aims to detect wheat diseases using a Convolutional Neural Network (CNN) model trained with TensorFlow/Keras. The model can classify images of wheat as either *Healthy Wheat* or affected by *Fusarium Head Blight*. A Flask web application is also provided to allow users to upload images and receive predictions in real-time.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [File Structure](#file-structure)
- [Model Training](#model-training)
- [Running the Flask Application](#running-the-flask-application)
- [Results](#results)
- [Contributing](#contributing)

## Project Overview
Wheat diseases can severely affect crop yield, and early detection is crucial for effective management. This project trains a CNN model on images of wheat to classify them as either healthy or diseased (Fusarium Head Blight). A simple web application is built using Flask to allow users to upload an image of wheat and receive a prediction.

### Goals:
- Develop a CNN-based model to detect wheat diseases.
- Build a web application that can take user input (wheat images) and predict the class of the wheat.
  
## Dataset
The dataset consists of two categories:
- **Healthy Wheat**
- **Fusarium Head Blight (Disease)**

The images are organized into subdirectories for each category and resized to 150x150 pixels for input to the CNN model. You can replace the dataset in the `wheat_dataset/` folder with your own images if needed.

## Model Architecture
The model is built using a Convolutional Neural Network (CNN) architecture, which includes:
- **Three Convolutional Layers** with MaxPooling.
- **Flattening** the 2D output into 1D.
- **Fully Connected Layers** with `ReLU` activation.
- **Output Layer** with `sigmoid` activation for binary classification (healthy or diseased).

### Layers:
1. **Conv2D**: 32 filters, (3,3) kernel, ReLU activation
2. **MaxPooling2D**: (2,2) pool size
3. **Conv2D**: 64 filters, (3,3) kernel, ReLU activation
4. **MaxPooling2D**: (2,2) pool size
5. **Conv2D**: 128 filters, (3,3) kernel, ReLU activation
6. **MaxPooling2D**: (2,2) pool size
7. **Flatten**: Converts 2D matrices into 1D.
8. **Dense**: 128 units, ReLU activation.
9. **Dropout**: 50% dropout for regularization.
10. **Dense**: 1 unit, `sigmoid` activation for binary output.

## Installation

### Prerequisites
Ensure you have the following installed on your system:
- Python 3.8
- TensorFlow
- Flask
- NumPy

### Clone the Repository
git clone https://github.com/ratnap02/Wheat-Disease-Detection.git
cd Wheat-Disease-Detection

## Usage
Model Training
The model is trained using the dataset located in the wheat_dataset/ folder. You can modify or add your dataset and train the model by running the training.ipynb notebook.

#### Running the Flask Application
1. After training the model, the Flask application can be used for real-time predictions.
2. Run the Flask app:
> python app.py 
3. Open your web browser and go to http://127.0.0.1:5000/.
4. Upload a wheat image to get the disease prediction. 