"""
create_sample_jds.py - Generates sample test PDFs (1 Resume + 2 Conflicting JDs)
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def create_pdf(filename: str, title: str, sections: list):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle('TStyle', parent=styles['Heading1'], fontSize=18, leading=22, spaceAfter=10)
    heading_style = ParagraphStyle('HStyle', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6)

    story.append(Paragraph(title, title_style))
    story.append(Spacer(1, 10))

    for heading, content in sections:
        story.append(Paragraph(heading, heading_style))
        story.append(Paragraph(content, body_style))

    doc.build(story)
    print(f"Generated sample PDF: {filename}")


def generate_all_samples():
    # 1. Sample Resume (Mid-level Data Scientist)
    resume_sections = [
        ("Candidate Overview", "Jane Doe - Data Scientist with 2.5 years of experience building machine learning models."),
        ("Technical Skills", "Python, PyTorch, Scikit-Learn, SQL, pandas, Git, Streamlit, Docker, REST APIs."),
        ("Work Experience", "Data Scientist at Tech Corp (2022 - Present): Developed predictive models using PyTorch and SQL. Built automated RAG pipelines using LangChain and ChromaDB. Deployed models via FastAPI."),
        ("Education", "B.S. in Computer Science, State University.")
    ]
    create_pdf("sample_resume.pdf", "Resume - Jane Doe (Data Scientist)", resume_sections)

    # 2. Sample JD 1 (Senior Data Scientist - Conflicting 5+ yrs requirement)
    jd_senior_sections = [
        ("Role Overview", "Senior Data Scientist (Role A)"),
        ("Experience Requirement", "Requires 5+ years of full-time industry experience in machine learning and data science."),
        ("Key Technical Skills", "PyTorch, Python, AWS SageMaker, Distributed Training, MLOps, SQL."),
        ("Responsibilities", "Lead AI architecture, deploy large-scale LLM pipelines, and mentor junior data scientists.")
    ]
    create_pdf("sample_jd_senior.pdf", "Job Description - Senior Data Scientist (Role A)", jd_senior_sections)

    # 3. Sample JD 2 (Junior Data Scientist - Conflicting 0-1 yrs requirement & Java skill)
    jd_junior_sections = [
        ("Role Overview", "Junior Data Scientist / Analyst (Role B)"),
        ("Experience Requirement", "Requires 0-1 years of experience. Fresh graduates welcome."),
        ("Key Technical Skills", "Java, SQL, Python, Excel, Basic Machine Learning."),
        ("Responsibilities", "Build dashboard analytics, write SQL queries, and assist in Java backend data pipelines.")
    ]
    create_pdf("sample_jd_junior.pdf", "Job Description - Junior Data Scientist (Role B)", jd_junior_sections)


if __name__ == "__main__":
    generate_all_samples()
