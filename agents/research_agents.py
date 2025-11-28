from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

genai.configure(api_key="AIzaSyD2-YvNVkV80VDjQ4aBLcnU4_INaBPhdN8")

llm = LLM(
    model="gemini/gemini-2.0-flash",
    api_key="AIzaSyD2-YvNVkV80VDjQ4aBLcnU4_INaBPhdN8"
)



# ----------------- Agents ----------------- #

pdf_agent = Agent(
    role="PDF Extraction Agent",
    goal="Extract clean text from the uploaded research paper PDF.",
    backstory="You are an expert in reading academic PDFs.",
    allow_delegation=False,
    llm=llm
)

summary_agent = Agent(
    role="Summarization Agent",
    goal="Generate a concise 150–200 word summary of the research paper.",
    backstory="You specialize in summarizing technical research papers accurately.",
    allow_delegation=False,
    llm=llm
)

keyword_agent = Agent(
    role="Keyword Extractor Agent",
    goal="Extract 5–10 important keywords from the research paper.",
    backstory="You identify important technical terms.",
    allow_delegation=False,
    llm=llm
)

reference_agent = Agent(
    role="Reference Extraction Agent",
    goal="Extract important references and citations from the research text.",
    backstory="You read and extract reference-style text.",
    allow_delegation=False,
    llm=llm
)

citation_agent = Agent(
    role="Citation Formatting Agent",
    goal="Generate APA, MLA, and IEEE formatted citations.",
    backstory="You are an expert in academic citation styles.",
    allow_delegation=False,
    llm=llm
)

def build_research_crew(text):

    tasks = [

        Task(
            name="keywords",
            description=f"Extract and  Write 5–10 important keywords:\n{text}",
            expected_output="List of keywords",
            agent=keyword_agent
        ),

        Task(
            name="summary",
            description=f"Write a 150–200 word summary:\n{text}",
            expected_output="Concise summary",
            agent=summary_agent
        ),

        Task(
            name="references",
            description=f"Extract and  Write references mentioned in:\n{text}",
            expected_output="Reference list",
            agent=reference_agent
        ),

        Task(
            name="citations",
            description=f"Generate APA, MLA, IEEE formatted citations.\n{text}",
            expected_output="3 formatted citations",
            agent=citation_agent
        )
    ]

    crew = Crew(
        agents=[summary_agent, keyword_agent, reference_agent, citation_agent],
        tasks=tasks,
        verbose=False  # Disable terminal logs
    )

    return crew
