"""FastAPI service for EuroSAT land-cover classification (fine-tuned ResNet18)."""

import io
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "models" / "resnet18_eurosat.pt"))
CLASS_NAMES_PATH = Path(os.getenv("CLASS_NAMES_PATH", BASE_DIR / "models" / "class_names.json"))

MODEL_NAME = "ResNet18 (fine-tuned)"
MODEL_VERSION = "1.0"

# Must match the preprocessing used for the transfer-learning model in the notebook.
IMG_SIZE = 128
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}

transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ]
)


class ClassScore(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    top_3: List[ClassScore]
    model: str
    version: str


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model: str
    version: str


def build_model(num_classes: int) -> nn.Module:
    """ResNet18 with the same classification head that was trained in the notebook."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
    return model


@lru_cache(maxsize=1)
def get_bundle():
    """Loads the trained model and class names once, on the first prediction request."""
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError(
            "Model files not found. Run the notebook to train the model, then make sure "
            f"{MODEL_PATH.name} and {CLASS_NAMES_PATH.name} are in the models/ folder."
        )
    with open(CLASS_NAMES_PATH, encoding="utf-8") as f:
        class_names = json.load(f)
    model = build_model(len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, class_names


def classify(image: Image.Image, model: nn.Module, class_names: List[str]) -> PredictionResponse:
    tensor = transform(image).unsqueeze(0)
    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]
    top = torch.topk(probs, k=min(3, len(class_names)))
    top_3 = [
        ClassScore(label=class_names[int(i)], probability=round(float(p), 4))
        for p, i in zip(top.values, top.indices)
    ]
    return PredictionResponse(
        predicted_class=top_3[0].label,
        confidence=top_3[0].probability,
        top_3=top_3,
        model=MODEL_NAME,
        version=MODEL_VERSION,
    )


app = FastAPI(
    title="EuroSAT Land-Cover Classifier API",
    description="Predicts the land-cover class of a satellite image patch using a fine-tuned ResNet18.",
    version=MODEL_VERSION,
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_available=MODEL_PATH.exists() and CLASS_NAMES_PATH.exists(),
        model=MODEL_NAME,
        version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")

    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="The file is not a valid image.")

    try:
        model, class_names = get_bundle()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    return await run_in_threadpool(classify, image, model, class_names)
