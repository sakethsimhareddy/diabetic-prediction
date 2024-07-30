from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import joblib
import os

# Load the model and scaler
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define the model path
model_path = os.path.join(base_dir, 'model.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)


feature_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 
    'BMI', 'DiabetesPedigreeFunction', 'Age'
]

@csrf_exempt
def index(request):
    if request.method == 'POST':
        features = [
            float(request.POST[f'feature_{i}']) for i in range(1, len(feature_columns) + 1)
        ]
        features = np.array(features).reshape(1, -1)
        features = scaler.transform(features)
        prediction = model.predict(features)
        return JsonResponse({'prediction': int(prediction[0])})
    
    context = {
        'feature_columns': feature_columns
    }
    return render(request, 'index.html', context)
