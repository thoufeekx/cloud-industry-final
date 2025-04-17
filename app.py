from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import os
from werkzeug.utils import secure_filename
import logging
import numpy as np
from typing import Any, Dict
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:8090", "http://127.0.0.1:8090"]}})

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
        credit_limit = convert_to_python_type(input_data[0])
        age = convert_to_python_type(input_data[1])
        bill_amount = convert_to_python_type(input_data[2])
        payment_amount = convert_to_python_type(input_data[3])
        
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

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        if model is None:
            load_model()
            if model is None:
                return jsonify({
                    'status': 'error',
                    'message': 'Model not loaded'
                }), 500
        return jsonify({
            'status': 'ok',
            'model_loaded': model is not None,
            'scaler_loaded': scaler is not None
        })
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Endpoint to make predictions"""
    global model, scaler
    
    try:
        # Get input data
        data = request.json.get('data')
        if not data:
            return jsonify({'error': 'No input data provided'}), 400

        # Convert to numpy array
        input_data = np.array([data])

        # Load model if not loaded
        if model is None:
            load_model()
            if model is None:
                return jsonify({'error': 'Model not loaded'}), 500

        # Scale the input data
        if scaler is None:
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

        scaled_data = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(scaled_data)[0]
        
        # Calculate risk factors
        risk_factors = calculate_risk_factors(input_data[0])
        
        # Convert numpy types to Python types
        risk_factors = {k: convert_to_python_type(v) for k, v in risk_factors.items()}
        
        return jsonify({
            'prediction': int(prediction),
            'risk_factors': risk_factors
        })

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model when starting the app
    try:
        load_model()
    except Exception as e:
        logger.error(f"Error loading model on startup: {str(e)}")
        raise

    app.run(debug=True)
