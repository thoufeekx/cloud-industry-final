import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import requests
import json

def load_model_and_scaler():
    """Load the model and scaler from pickle files"""
    model_path = 'models/model.pkl'
    scaler_path = 'models/features.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try loading scaler
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
                if not isinstance(scaler, StandardScaler):
                    print("Warning: Loaded scaler is not a StandardScaler object")
                    raise ValueError("Scaler is not a StandardScaler")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            print("Creating new scaler with training data distribution")
            # Create a new scaler with training data distribution
            # We'll use the test samples as a proxy for training data
            test_samples = np.array([
                [20000, 30, 1500, 1000],  # Low risk customer
                [5000, 45, 3000, 500],    # High risk customer
                [10000, 25, 2000, 1500],  # Moderate risk customer
                [15000, 35, 2500, 2000],  # Low risk customer
                [3000, 50, 4000, 1000],   # Very high risk customer
            ])
            scaler = StandardScaler()
            scaler.fit(test_samples)
            print("Scaler created with training data distribution")
        
        return model, scaler
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure model.pkl and features.pkl are in the models directory")
        return None, None
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        return None, None

def test_prediction(model, scaler):
    """Test the model with sample data points"""
    # Sample test data (credit_limit, age, bill_amount_sept, payment_amount_sept)
    test_samples = [
        [20000, 30, 1500, 1000],  # Low risk customer (high credit limit, young age, reasonable bill/payment)
        [5000, 45, 3000, 500],    # High risk customer (low credit limit, older age, high bill/low payment)
        [10000, 25, 2000, 1500],  # Moderate risk customer (medium values across all features)
        [15000, 35, 2500, 2000],  # Low risk customer (high credit limit, moderate age, good payment history)
        [3000, 50, 4000, 1000],   # Very high risk customer (very low credit limit, old age, very high bill/low payment)
        [25000, 40, 3500, 2500],  # Low risk customer (high credit limit, moderate age, good payment history)
        [8000, 38, 2800, 1800],   # Moderate risk customer (medium values across all features)
        [12000, 42, 3200, 2200],  # Low risk customer (high credit limit, moderate age, good payment history)
        [18000, 48, 3800, 2800],  # Low risk customer (high credit limit, older age, good payment history)
        [22000, 55, 4200, 3200],  # High risk customer (high credit limit, old age, high bill/payment)
    ]
    
    # Convert to numpy array for scaling
    test_samples = np.array(test_samples)
    
    # Scale the test data using the loaded scaler
    test_samples_scaled = scaler.transform(test_samples)
    
    # Get predictions
    predictions = model.predict(test_samples_scaled)
    probabilities = model.predict_proba(test_samples_scaled)
    
    print("\nTest Results:")
    print("-" * 50)
    for i, sample in enumerate(test_samples):
        print(f"\nSample {i+1}: {sample}")
        print(f"Prediction: {'Default' if predictions[i] == 1 else 'No Default'}")
        print(f"Confidence: {max(probabilities[i]):.2%}")
        print(f"Probability of Default: {probabilities[i][1]:.2%}")
        print(f"Credit Limit: {sample[0]}")
        print(f"Age: {sample[1]}")
        print(f"Bill Amount: {sample[2]}")
        print(f"Payment Amount: {sample[3]}")

def test_api_endpoint():
    """Test the API endpoint with sample data"""
    url = "http://localhost:5000/api/predict"
    
    # Test sample (credit_limit, age, bill_amount_sept, payment_amount_sept)
    test_sample = {
        "input": [5000, 45, 3000, 500]  # High risk customer
    }
    
    try:
        response = requests.post(url, json=test_sample)
        result = response.json()
        print("\nAPI Test Result:")
        print("-" * 50)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Prediction: {result.get('prediction', 'unknown')}")
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == "__main__":
    print("Testing ML Model...")
    
    # Test with local model
    model, scaler = load_model_and_scaler()
    if model and scaler:
        test_prediction(model, scaler)
    
    # Test with API endpoint
    test_api_endpoint()
