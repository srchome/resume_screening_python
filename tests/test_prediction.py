# tests/test_prediction.py

from model_training.model_utils import load_model
import os

def test_model_load_and_predict_valid_score():
    model_path = "model_training/models/resume_model.joblib"
    assert os.path.exists(model_path), f"Model not found at {model_path}"

    model = load_model(model_path)

    # Provide a dummy input similar to what the pipeline expects
    input_data = {
        "ResumeText": "Experienced software engineer with expertise in Python, Django, and REST APIs.",
        "JobTitle": "Software Engineer"
    }

    prediction = model.predict([input_data])[0]

    # Ridge regression can return scores beyond 0-1, so we check for reasonable range
    assert isinstance(prediction, float), "Prediction is not a float"
    assert -1.0 <= prediction <= 2.0, f"Prediction out of expected range: {prediction}"

def test_model_with_empty_input():
    model = load_model("model_training/models/resume_model.joblib")

    input_data = {
        "ResumeText": "",
        "JobTitle": ""
    }

    prediction = model.predict([input_data])[0]
    assert isinstance(prediction, float), "Prediction is not a float"
