# waste-classifier-api
A FastAPI-powered deep learning application that classifies waste images into categories (e.g., cardboard, plastic, metal, glass, paper, trash) and deploys using Docker &amp; Kubernetes (Minikube) for scalable inference.

# Waste Classifier API

A deep learning-powered API built with FastAPI that classifies waste images (e.g., cardboard, plastic, metal, glass, paper, trash) to assist in smart waste management. The application is trained using a CNN model in TensorFlow/Keras and deployed locally using Docker and Kubernetes (via Minikube).

## Features

-  Trained CNN model using image dataset
-  FastAPI backend for serving predictions
-  Upload and classify waste images
-  Dockerized application
-  Kubernetes deployment with Minikube support
-  REST endpoint: `/predict`

## Model Details

- Architecture: Convolutional Neural Network (CNN)
- Framework: TensorFlow / Keras
- Trained using `ImageDataGenerator` with augmentation
- Classes: e.g., Cardboard, Plastic, Glass, Metal, Paper, Trash.
- Input size: 150x150 RGB

## Project Structure

waste_classifier_app/
├── app/
│ └── main.py # FastAPI app
├── model/
│ ├── waste_classifier.h5 # Trained model
│ └── class_names.txt # Class label names
├── train_model.py # Script to train the CNN model
├── deployment.yaml # Kubernetes Deployment manifest
├── service.yaml # Kubernetes Service manifest
├── Dockerfile # Docker container setup
├── requirements.txt # Python dependencies
└── README.md 

## Setup Instructions

### Prerequisites

- Python 3.8+
- Docker
- Minikube (with Docker driver)
- kubectl
- Git

### Step 1: Clone and Setup Environment

git clone https://github.com/RamyaSP6/waste-classifier-api.git
cd waste-classifier-api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Step 2: Train the Model
python train_model.py
This will:

Load the dataset

Train the CNN model

Save it to model/waste_classifier.h5 and class_names.txt

Step 3: Run API Locally
uvicorn app.main:app --reload
Visit http://localhost:8000/docs to try it out.

Step 4: Build Docker Image
docker build -t waste-classifier-api .
Step 5: Deploy with Minikube
minikube start --driver=docker
eval $(minikube docker-env)
docker build -t waste-classifier-api .
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
Step 6: Access API
minikube service waste-api-service
Open the given 127.0.0.1:<PORT>/docs link to upload and classify images.

API Endpoint
POST /predict
Request: Upload image file

Response:

json
Copy
Edit
{
  "predicted_class": "glass",
  "confidence": 0.93
}
