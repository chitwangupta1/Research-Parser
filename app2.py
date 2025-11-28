import os
from flask import Flask, render_template, request
import PyPDF2
from dotenv import load_dotenv
from agents.research_agents import build_research_crew

load_dotenv()

app = Flask(__name__)


# ----------------------- PDF Extraction Function ----------------------- #

def extract_pdf_text(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        try:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        except:
            pass

    return text.strip()


# ---------------------------- Flask Routes ----------------------------- #

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        pdf = request.files.get("pdf_file")

        if not pdf or pdf.filename == "":
            return render_template("index.html", error="Please upload a PDF file.")

        # Step 1: Extract PDF text
        extracted_text = extract_pdf_text(pdf)

        if extracted_text.strip() == "":
            return render_template("index.html", error="Unable to extract text from PDF.")

        # Step 2: Create Agent Crew
        crew = build_research_crew(extracted_text)

        # Step 3: Run CrewAI pipeline
        result_output = crew.kickoff()

        # result_output is a CrewOutput object → convert to text
        final_output = result_output.raw

        return render_template("index.html", result=final_output)

    return render_template("index.html")


# ---------------------------- Run Flask App ---------------------------- #

if __name__ == "__main__":
    app.run(debug=True)
