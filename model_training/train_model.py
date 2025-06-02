# train_model.py

from data_loader import load_training_data
from model_utils import build_model_pipeline, save_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import argparse
from sklearn.metrics import mean_squared_error
import numpy as np

def main(data_path: str):
    print("📄 Loading data...")
    df = load_training_data(data_path)

    X = df[["ResumeText", "JobTitle"]]
    y = df["Label"]

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🏗️ Building pipeline...")
    pipeline = build_model_pipeline()

    print("🧠 Training model...")
    pipeline.fit(X_train, y_train)

    print("✅ Model trained!")

    print("📊 Evaluating model...")
    y_pred = pipeline.predict(X_test)
    #print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    print("💾 Saving model...")
    save_model(pipeline)
    print("✅ Model saved as 'resume_model.joblib'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train resume screening ML model.")
    parser.add_argument("--data", type=str, required=True, help="Path to training_data.csv")
    args = parser.parse_args()
    main(args.data)
