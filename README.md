# MNIST Digit Prediction Pipeline

This project implements a pipeline for MNIST digit prediction using a CNN, FastAPI, and Streamlit.

## Project Structure

```
├── code
│   ├── datasets
│   ├── deployment
│   │   ├── api/          # FastAPI service
│   │   └── app/          # Streamlit web app
│   └── models/           # Model training code
├── data/                 # MNIST dataset (downloaded automatically)
├── models/               # Trained model files
└── README.md
```

## Quick Start

### Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local training)

### 1. Train the Model

First, you need to train the model. You can do this locally:

```bash
python train_model.py
```

### 2. Deploy with Docker Compose

```bash
cd code/deployment
docker-compose up --build
```

This will:
- Build both API and Streamlit app containers
- Start the FastAPI service on port 8000
- Start the Streamlit app on port 8501
- Create a network for service communication

### 3. Access the Application

- **Web App**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs
- **API Health Check**: http://localhost:8000/health

## Usage

1. Open the Streamlit app in your browser
2. Draw a digit (0-9) in the canvas using your mouse or touch
3. Click "Predict" to get the AI prediction
4. View the predicted digit and confidence score
5. Use "Clear" to start over
