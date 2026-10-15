# EuroSAT Land-Cover Classification with CNNs and FastAPI

An end-to-end deep learning project that classifies satellite image patches into 10 land-use / land-cover classes and serves the trained model through a FastAPI REST API.

The project covers data exploration, a stratified train/validation/test split, a CNN trained from scratch, transfer learning with ResNet18, evaluation with a confusion matrix, Grad-CAM explainability, model serialization, API serving, and automated tests.

---

## Dataset

[EuroSAT](https://github.com/phelber/EuroSAT) is a public dataset of Sentinel-2 satellite image patches covering 10 land-cover classes (for example forest, river, highway, residential and sea/lake). The notebook downloads it automatically through `torchvision.datasets.EuroSAT`.

The data is split 70% / 15% / 15% into train, validation and test sets, stratified by class.

---

## Approach

1. **Baseline CNN** trained from scratch: four Conv-BatchNorm-ReLU-MaxPool blocks, global average pooling and dropout.
2. **Transfer learning:** a ResNet18 pretrained on ImageNet, fine-tuned on EuroSAT with a lower learning rate for the backbone than for the new classification head. Images are upscaled to 128x128.
3. **Evaluation** on the untouched test set: accuracy, per-class precision/recall/F1, confusion matrix and the most-confused class pairs.
4. **Explainability:** Grad-CAM heatmaps from the last convolutional block show which regions drive each prediction.

---

## Results

Run `notebooks/EuroSAT_Land_Cover_CNN.ipynb` and fill in the table with the values it prints.

| Model | Parameters | Test accuracy |
|---|---:|---:|
| Baseline CNN (from scratch) | _add_ | _add_ |
| ResNet18 (fine-tuned) | _add_ | _add_ |

Add the confusion matrix and a few Grad-CAM examples from the notebook here, with a short note on which classes the model confuses and why.

---

## API

The API loads the fine-tuned ResNet18 and exposes two endpoints.

### Health check

```http
GET /health
```

### Prediction

```http
POST /predict
```

Upload a JPEG or PNG image as multipart form data (maximum 5 MB):

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@tile.png;type=image/png"
```

Response shape (values below are illustrative):

```json
{
  "predicted_class": "Forest",
  "confidence": 0.97,
  "top_3": [
    {"label": "Forest", "probability": 0.97},
    {"label": "HerbaceousVegetation", "probability": 0.02},
    {"label": "Pasture", "probability": 0.01}
  ],
  "model": "ResNet18 (fine-tuned)",
  "version": "1.0"
}
```

Error responses: `415` for a non-JPEG/PNG upload, `400` for an empty or corrupt image, `413` for a file over 5 MB, and `503` if the trained model files are missing.

Interactive documentation is available at `http://127.0.0.1:8000/docs` while the server is running.

---

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/Dwivedisandeep1/eurosat-land-cover-cnn.git
cd eurosat-land-cover-cnn
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Train the model

Open `notebooks/EuroSAT_Land_Cover_CNN.ipynb` and run all cells (a GPU, for example in Google Colab, makes training much faster). This writes the trained model to the `models/` folder:

```text
models/
├── resnet18_eurosat.pt
├── simple_cnn.pt
└── class_names.json
```

If you trained on Colab, download `resnet18_eurosat.pt` and `class_names.json` into the `models/` folder of your local copy.

### 4. Start the API

```bash
uvicorn api.main:app --reload
```

### 5. Run the tests

```bash
pytest -q
```

The tests use an untrained model, so they run without the trained weights.

---

## Project structure

```text
eurosat-land-cover-cnn/
├── api/
│   ├── __init__.py
│   └── main.py
├── models/
│   └── .gitkeep
├── notebooks/
│   └── EuroSAT_Land_Cover_CNN.ipynb
├── tests/
│   └── test_api.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Tech stack

- Python
- PyTorch and torchvision
- scikit-learn, pandas, NumPy
- Matplotlib and Seaborn
- FastAPI, Uvicorn and Pydantic
- Pytest

---

## Limitations

- EuroSAT patches are 64x64 pixels and cover European locations, so accuracy may drop on imagery from other regions or sensors.
- The API classifies one patch at a time; it does not segment large scenes.
- Grad-CAM shows where the model looked, not why a class is correct.

---

## Future improvements

- Compare more architectures (MobileNetV3, EfficientNet) on accuracy versus model size
- Test generalization on imagery from another region
- Containerize the API with Docker
- Add batch prediction and CI for the tests

---

## Author

**Sandeep Dwivedi**

GitHub: https://github.com/Dwivedisandeep1

---

## Disclaimer

This is a portfolio and learning project. Predictions are estimates from the training data and should not be used for operational decisions.
