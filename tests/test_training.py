# tests/test_training.py

import os
import sys
import pandas as pd
import tempfile
from unittest.mock import patch

# Add model_training to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../model_training')))

import train_model
import model_utils

def test_training_pipeline_without_affecting_main_model():
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_csv = os.path.join(tmpdir, "training_data.csv")
        temp_model_path = os.path.join(tmpdir, "temp_model.joblib")

        # Create mock training data
        df = pd.DataFrame([
            {"ResumeText": "Python ML engineer", "JobTitle": "ML Engineer", "Label": 0.9},
            {"ResumeText": "Marketing and sales", "JobTitle": "Sales", "Label": 0.2}
        ])
        df.to_csv(temp_csv, index=False)

        # Patch save_model INSIDE train_model module
        with patch('train_model.save_model') as mock_save:
            def save_model_mock(model, path="models/resume_model.joblib"):
                model_utils.save_model(model, temp_model_path)

            mock_save.side_effect = save_model_mock

            # Run training
            train_model.main(temp_csv)

        # Assert temp model created
        assert os.path.exists(temp_model_path), "Temp model not created."

        # Load and test prediction
        model = model_utils.load_model(temp_model_path)
        test_input = pd.DataFrame([{
            "ResumeText": "Experienced software engineer",
            "JobTitle": "Software Developer"
        }])
        prediction = model.predict(test_input)[0]
        assert isinstance(prediction, float)
