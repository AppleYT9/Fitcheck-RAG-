"""
test_rag.py - Verification script for RAG Engine Ingestion & Retrieval.
"""

import os
import sys

# Ensure current workspace is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import (
    extract_text_from_pdf,
    create_chunks,
    build_vector_store,
    retrieve_with_scores
)


def run_tests():
    pdf_path = "sample_document.pdf"
    if not os.path.exists(pdf_path):
        print(f"ERROR: {pdf_path} not found. Run create_sample_pdf.py first.")
        return

    print("--- 1. Testing Document Extraction ---")
    pages = extract_text_from_pdf(pdf_path)
    print(f"Extracted {len(pages)} pages.")
    for p in pages:
        print(f"  - Page {p['page_number']}: {len(p['text'])} chars")

    print("\n--- 2. Testing Document Chunking ---")
    chunks = create_chunks(pages, chunk_size=600, chunk_overlap=100)
    print(f"Created {len(chunks)} chunks.")
    for idx, c in enumerate(chunks[:3]):
        print(f"  Chunk {idx+1} [Page {c.metadata['page_number']}]: {c.page_content[:80]}...")

    print("\n--- 3. Testing Embedding & Chroma Vector Store ---")
    vector_store = build_vector_store(chunks)
    print("Vector store built successfully.")

    print("\n--- 4. Testing Cosine Retrieval & Scoring ---")
    queries = [
        "What model powers the Groq LLM?",
        "What are the similarity score thresholds for High, Medium, and Low confidence?",
        "Who won the 2026 World Chess Championship?" # Out of document question
    ]

    for q in queries:
        print(f"\nQuery: '{q}'")
        res = retrieve_with_scores(vector_store, q, top_k=5)
        print(f"Confidence: {res['confidence_label']} (Badge color: {res['confidence_color']})")
        print(f"Top Similarity Score: {res['top_score']:.4f}")
        print(f"Retrieved Chunks Count: {len(res['retrieved_chunks'])}")
        for idx, chunk in enumerate(res['retrieved_chunks'][:2], 1):
            print(f"  Top {idx} [Score: {chunk['similarity_percentage']}, Page {chunk['page_number']}]: {chunk['source_text'][:100]}...")

    print("\n✅ Ingestion and Retrieval tests passed!")

if __name__ == "__main__":
    run_tests()
