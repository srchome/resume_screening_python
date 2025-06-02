# data_loader.py
import pandas as pd

def load_training_data(filepath: str) -> pd.DataFrame:
    """
    Load and validate training data.
    
    Parameters:
        filepath (str): Path to the training CSV file.
    
    Returns:
        pd.DataFrame: Cleaned training data.
    """
    df = pd.read_csv(filepath)

    required_columns = ['ResumeText', 'JobTitle', 'Label']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Optional cleanup
    df = df.dropna(subset=required_columns)
    
    return df
