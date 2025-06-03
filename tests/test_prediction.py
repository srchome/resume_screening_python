# tests/test_prediction.py

import os
import sys
import pandas as pd
from model_training.model_utils import load_model

# Ensure model_training module is discoverable
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_training')))  
  
def test_model_load_and_predict_valid_score():
    model_path = "model_training/models/resume_model.joblib"
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    model = load_model(model_path)

    input_df = pd.DataFrame([{
        "ResumeText": "Experienced software engineer with expertise in Python, Django, and REST APIs.",
        "JobTitle": "Software Engineer"
    }])

    prediction = model.predict(input_df)[0]

    assert isinstance(prediction, float), "Prediction is not a float"
    assert -1.0 <= prediction <= 2.0, f"Prediction out of expected range: {prediction}"

def test_model_with_empty_input():
    model = load_model("model_training/models/resume_model.joblib")

    input_df = pd.DataFrame([{
        "ResumeText": "",
        "JobTitle": ""
    }])

    prediction = model.predict(input_df)[0]

    assert isinstance(prediction, float), "Prediction is not a float"
