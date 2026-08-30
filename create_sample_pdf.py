"""
create_sample_pdf.py - Helper script to generate a sample PDF document for testing.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_sample_pdf(filename: str = "sample_document.pdf"):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle(
        'TitleStyle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'HeadingStyle',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyStyle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8
    )

    # Page 1 Content
    story.append(Paragraph("Explainable Document Q&A System - Sample Report", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("1. Executive Summary", heading_style))
    story.append(Paragraph(
        "Retrieval-Augmented Generation (RAG) combines information retrieval with large language models (LLMs) "
        "to answer questions based on ground-truth documents. This application, 'Explainable Document Q&A', "
        "is built using Python 3.11, Streamlit, LangChain, ChromaDB, sentence-transformers, and the Groq LLM API.",
        body_style
    ))
    story.append(Paragraph("2. Technical Specifications & Stack", heading_style))
    story.append(Paragraph(
        "<b>User Interface:</b> Streamlit provides an interactive chat interface and document upload panel.<br/>"
        "<b>Orchestration:</b> LangChain manages document loading, text splitting, and retrieval pipelines.<br/>"
        "<b>Vector Store:</b> ChromaDB runs locally in-memory, storing dense vector representations.<br/>"
        "<b>Embeddings:</b> The model 'sentence-transformers/all-MiniLM-L6-v2' generates 384-dimensional embeddings.<br/>"
        "<b>LLM Provider:</b> Groq API powers the 'llama-3.3-70b-versatile' model for fast, context-bounded generation.",
        body_style
    ))
    story.append(Paragraph("3. Ingestion and Processing Pipeline", heading_style))
    story.append(Paragraph(
        "Documents are parsed page-by-page using pypdf. Text is chunked with LangChain's RecursiveCharacterTextSplitter "
        "configured with a chunk_size of 600 characters and a chunk_overlap of 100 characters. "
        "Each chunk preserves metadata including chunk_id, page_number, and source_text.",
        body_style
    ))

    story.append(PageBreak())

    # Page 2 Content
    story.append(Paragraph("4. Explainability Layer & Confidence Scoring", heading_style))
    story.append(Paragraph(
        "A key differentiator of this system is the Explainability Layer. For every user query, the vector store "
        "retrieves the top-5 chunks via cosine similarity.<br/>"
        "Confidence is assigned based on the top chunk similarity score:<br/>"
        "• <b>High Confidence:</b> Similarity score above 0.5 (Green badge).<br/>"
        "• <b>Medium Confidence:</b> Similarity score between 0.3 and 0.5 (Yellow badge).<br/>"
        "• <b>Low Confidence:</b> Similarity score below 0.3 (Red badge — answer may be unreliable).",
        body_style
    ))
    story.append(Paragraph("5. Security, Environment, and Constraints", heading_style))
    story.append(Paragraph(
        "API keys are strictly loaded from environment variables (GROQ_API_KEY) and are never hardcoded. "
        "If a scanned PDF without extractable text is uploaded, the ingestion pipeline handles it gracefully "
        "by throwing an explicit user error.",
        body_style
    ))

    doc.build(story)
    print(f"Sample PDF successfully created at: {filename}")


if __name__ == "__main__":
    generate_sample_pdf()
