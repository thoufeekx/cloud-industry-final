import requests
import json

# Test profiles with different risk levels
test_profiles = [
    # Low risk customer
    {
        'input': [20000, 30, 1500, 1000],
        'expected': 'Low Risk'
    },
    # High risk customer
    {
        'input': [5000, 45, 3000, 500],
        'expected': 'High Risk'
    },
    # Moderate risk customer
    {
        'input': [10000, 35, 2000, 1500],
        'expected': 'Moderate Risk'
    },
    # Very high risk customer
    {
        'input': [3000, 50, 4000, 1000],
        'expected': 'Very High Risk'
    },
    # High credit limit but poor payment history
    {
        'input': [25000, 40, 3500, 2500],
        'expected': 'Low Risk'
    },
    # Young customer with good payment history
    {
        'input': [8000, 25, 2000, 1800],
        'expected': 'Low Risk'
    }
]

# base_url = 'http://localhost:5000/api/predict'
base_url = 'https://credit-risk-backend.azurewebsites.net/api/predict'

print("Testing different customer profiles:")
print("-" * 50)

for i, profile in enumerate(test_profiles, 1):
    print(f"\nTest Profile {i}:")
    print(f"Input: {profile['input']}")
    print(f"Expected Risk: {profile['expected']}")
    
    response = requests.post(
        base_url,
        headers={'Content-Type': 'application/json'},
        json={'input': profile['input']}
    )
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(f"Error message: {response.text}")
        continue
    
    data = response.json()
    print("API Response:")
    print(json.dumps(data, indent=2))
    
    prediction = data.get('prediction', [None])[0]
    probability = data.get('probability', [None])[0]
    
    if prediction is None or probability is None:
        print("Error: Missing prediction or probability in response")
        continue
    
    print(f"\nPrediction: {'Default' if prediction == 1 else 'No Default'}")
    print(f"Probability of Default: {probability * 100:.1f}%")
    print(f"Confidence: {(1 - probability) * 100:.1f}%")
    
    # Determine actual risk level based on our thresholds
    if prediction == 0:  # No Default
        if probability < 0.2:
            actual_risk = "Low Risk"
        elif probability < 0.4:
            actual_risk = "Moderate Risk"
        elif probability < 0.6:
            actual_risk = "High Risk"
        else:
            actual_risk = "Very High Risk"
    else:  # Default
        if probability > 0.8:
            actual_risk = "Very High Risk"
        elif probability > 0.6:
            actual_risk = "High Risk"
        elif probability > 0.4:
            actual_risk = "Moderate Risk"
        else:
            actual_risk = "Low Risk"
    
    print(f"\nActual Risk Level: {actual_risk}")
    print(f"Expected Risk Level: {profile['expected']}")
    print("-" * 50)
