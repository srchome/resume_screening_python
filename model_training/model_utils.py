# model_utils.py

import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# These are the named functions replacing lambdas
def extract_resume(x):
    return x["ResumeText"]

def extract_jobtitle(x):
    return x["JobTitle"]

def build_model_pipeline() -> Pipeline:
    resume_pipeline = Pipeline([
        ("extract", FunctionTransformer(extract_resume, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=5000))
    ])

    jobtitle_pipeline = Pipeline([
        ("extract", FunctionTransformer(extract_jobtitle, validate=False)),
        ("tfidf", TfidfVectorizer(max_features=500))
    ])

    full_pipeline = Pipeline([
        ("features", FeatureUnion([
            ("resume", resume_pipeline),
            ("jobtitle", jobtitle_pipeline)
        ])),
        ("regressor", Ridge())
    ])

    return full_pipeline

def save_model(model: Pipeline, path: str = "models/resume_model.joblib"):
    joblib.dump(model, path)

def load_model(path: str = "models/resume_model.joblib") -> Pipeline:
    return joblib.load(path)
