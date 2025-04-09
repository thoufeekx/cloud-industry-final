from flask import Flask, request, jsonify
import pickle
import os
from werkzeug.utils import secure_filename
import logging
import numpy as np
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create models directory if it doesn't exist
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

# Configuration
ALLOWED_EXTENSIONS = {'pkl'}

# Global variables for model and features
model = None
scaler = None

def load_model():
    """Load the model and scaler from pickle files"""
    global model, scaler
    try:
        model_path = os.path.join(MODELS_DIR, 'model.pkl')
        scaler_path = os.path.join(MODELS_DIR, 'features.pkl')
        
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            logger.info("Model loaded successfully")
        
        if os.path.exists(scaler_path):
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                if not isinstance(scaler, StandardScaler):
                    logger.warning("Loaded scaler is not a StandardScaler object")
                    raise ValueError("Scaler is not a StandardScaler")
                logger.info("Scaler loaded successfully")
            except Exception as e:
                logger.error(f"Error loading scaler: {str(e)}")
                logger.info("Creating new scaler with training data distribution")
                # Create a new scaler with training data distribution
                test_samples = np.array([
                    [20000, 30, 1500, 1000],  # Low risk customer
                    [5000, 45, 3000, 500],    # High risk customer
                    [10000, 25, 2000, 1500],  # Moderate risk customer
                    [15000, 35, 2500, 2000],  # Low risk customer
                    [3000, 50, 4000, 1000],   # Very high risk customer
                ])
                scaler = StandardScaler()
                scaler.fit(test_samples)
                logger.info("Scaler created with training data distribution")
        else:
            logger.info("Scaler file not found, creating new scaler")
            # Create a new scaler with training data distribution
            test_samples = np.array([
                [20000, 30, 1500, 1000],  # Low risk customer
                [5000, 45, 3000, 500],    # High risk customer
                [10000, 25, 2000, 1500],  # Moderate risk customer
                [15000, 35, 2500, 2000],  # Low risk customer
                [3000, 50, 4000, 1000],   # Very high risk customer
            ])
            scaler = StandardScaler()
            scaler.fit(test_samples)
            logger.info("Scaler created with training data distribution")
            
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def convert_to_python_type(value: Any) -> Any:
    """Convert numpy types to Python types for JSON serialization"""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    return value

def calculate_risk_factors(input_data: np.ndarray) -> Dict[str, Any]:
    """Calculate risk factors from input data"""
    try:
        credit_limit = convert_to_python_type(input_data[0][0])
        age = convert_to_python_type(input_data[0][1])
        bill_amount = convert_to_python_type(input_data[0][2])
        payment_amount = convert_to_python_type(input_data[0][3])
        
        # Calculate risk factors
        payment_ratio = payment_amount / bill_amount if bill_amount > 0 else 1.0
        credit_utilization = bill_amount / credit_limit if credit_limit > 0 else 1.0
        
        return {
            'payment_ratio': float(payment_ratio),
            'credit_utilization': float(credit_utilization),
            'age': int(age),
            'credit_limit': int(credit_limit)
        }
    except Exception as e:
        logger.error(f"Error calculating risk factors: {str(e)}")
        return {}

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        if not data or 'input' not in data:
            return jsonify({'error': 'Input data is required'}), 400
            
        # Preprocess input data
        input_data = np.array([data['input']])
        
        # Scale the input data
        if scaler is not None:
            input_data = scaler.transform(input_data)
        
        # Get both prediction and probability scores
        prediction = model.predict(input_data)[0]  # Get single prediction
        probability = model.predict_proba(input_data)[0][1]  # Get probability of default (class 1)
        
        # Convert numpy types to regular Python types
        prediction = convert_to_python_type(prediction)
        probability = convert_to_python_type(probability)
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(input_data)
        
        response = {
            'prediction': [prediction],
            'probability': [probability],
            'status': 'success',
            'risk_factors': risk_factors
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model when starting the app
    try:
        load_model()
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        raise

    app.run(debug=True)
