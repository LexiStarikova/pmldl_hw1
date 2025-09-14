from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import io
import joblib
import numpy as np
import uvicorn
from pathlib import Path

import sys

sys.path.append("/app/models")
from train_model import SimpleCNN

app = FastAPI(title="MNIST Digit Prediction API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
transform = None


def load_model():
    """Load the trained model and preprocessing transform"""
    global model, transform

    try:
        model_info_path = Path("/app/models/model_info.pkl")
        if not model_info_path.exists():
            raise FileNotFoundError("Model info file not found")

        model_info = joblib.load(model_info_path)

        model = SimpleCNN(num_classes=model_info["num_classes"])

        model_path = Path("/app/models/mnist_cnn.pth")
        if not model_path.exists():
            raise FileNotFoundError("Model weights file not found")

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize(
                    (model_info["transform_mean"],), (model_info["transform_std"],)
                ),
            ]
        )

        print("Model loaded successfully")

    except Exception as e:
        print(f"Error loading model: {e}")
        raise e


@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    load_model()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "MNIST Digit Prediction API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        img_array = np.array(image)
        if len(img_array.shape) == 3:
            img_array = np.mean(img_array, axis=2)

        # invert colors: white background -> black, black drawing -> white
        img_array = 255 - img_array

        image = Image.fromarray(img_array.astype(np.uint8), mode="L")

        input_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(outputs, dim=1).item()
            confidence = probabilities[0][predicted_class].item()

        # get top 3 predictions
        top3_probs, top3_indices = torch.topk(probabilities[0], 3)
        top3_predictions = [
            {"digit": int(idx.item()), "confidence": float(prob.item())}
            for idx, prob in zip(top3_indices, top3_probs)
        ]

        prediction_result = {
            "predicted_digit": predicted_class,
            "confidence": confidence,
            "top3_predictions": top3_predictions,
        }

        return prediction_result

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
