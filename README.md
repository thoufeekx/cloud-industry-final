# ML Model Backend

This backend service serves ML model predictions.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your model files in the `models` directory:
- `model.pkl` - Your trained model
- `features.pkl` - Feature scaler/transformer (if applicable)

3. Run the server:
```bash
python app.py
```

## API Endpoints

### Upload Model
```
POST /api/upload-model
```
Upload both model.pkl and features.pkl files

### Make Prediction
```
POST /api/predict
```
Send JSON data with input features
