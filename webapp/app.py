# webapp/app.py

from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import joblib
import tempfile
import PyPDF2
from docx import Document
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model_training')))


app = Flask(__name__)
model = joblib.load(os.path.join("model_training", "models", "resume_model.joblib"))

ALLOWED_EXTENSIONS = {"pdf", "docx"}
results = []
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file):
    ext = file.filename.rsplit('.', 1)[1].lower()
    if ext == "pdf":
        reader = PyPDF2.PdfReader(file)
        return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    elif ext == "docx":
        doc = Document(file)
        return " ".join([para.text for para in doc.paragraphs])
    return ""

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_title = request.form.get("job_title")
        job_desc = request.form.get("job_description")
        resumes = request.files.getlist("resumes")
        #results = []

        for resume in resumes:
            if resume and allowed_file(resume.filename):
                text = extract_text(resume)
                sample = pd.DataFrame([{"ResumeText": text, "JobTitle": job_title}])
                score = model.predict(sample)[0]

                results.append({"filename": resume.filename, "score": score})

        return render_template("results.html", results=results)

    return render_template("index.html")

@app.route("/privacy")
def privacy():
    return render_template("privacy.html")

@app.route('/download')
def download_csv():
    from flask import Response
    import csv
    from io import StringIO

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Filename', 'Score'])

    for result in results:  # Ensure 'results' is available
        writer.writerow([result['filename'], result['score']])

    output.seek(0)
    return Response(output, mimetype='text/csv',
                    headers={'Content-Disposition': 'attachment;filename=screening_results.csv'})


if __name__ == "__main__":
    app.run(debug=True)
