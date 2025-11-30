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

        # Raw text from all tasks
        # raw_output = result_output.raw

        result_output = crew.kickoff()

        # Each task result is stored in result_output.tasks_output
        summary    = result_output.tasks_output[1]  # first task: keywords or summary
        keywords   = result_output.tasks_output[0]  # adjust order according to task list
        references = result_output.tasks_output[2]
        citations  = result_output.tasks_output[3]

        return render_template(
            "index.html",
            keywords=keywords,
            summary=summary,
            references=references,
            citations=citations
)
        
    return render_template("index.html")


# ---------------------------- Run Flask App ---------------------------- #

# if __name__ == "__main__":
#     app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))  # Render sets PORT automatically
    app.run(host="0.0.0.0", port=port)


