from model_training import model_utils

def test_extract_resume_valid():
    resume = {"ResumeText": "Skilled in Python and Flask."}
    text = model_utils.extract_resume(resume)
    assert isinstance(text, str)
    assert "Python" in text
