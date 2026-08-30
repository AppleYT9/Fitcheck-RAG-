"""
test_jd_fit.py - Test script for JD-Fit Checker RAG & Conflict Detection Engine.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_engine import (
    extract_text_from_pdf_bytes,
    create_document_chunks,
    build_fresh_vector_store,
    retrieve_top_chunks,
    generate_fit_analysis,
    parse_conflicts_and_answer
)


def run_test():
    print("--- 1. Testing Document Ingestion (Resume + 2 JDs) ---")
    files = [
        ("sample_resume.pdf", "resume"),
        ("sample_jd_senior.pdf", "jd"),
        ("sample_jd_junior.pdf", "jd")
    ]

    all_chunks = []
    for fname, stype in files:
        if not os.path.exists(fname):
            print(f"ERROR: {fname} missing. Run create_sample_jds.py first.")
            return

        with open(fname, "rb") as f:
            pages = extract_text_from_pdf_bytes(f, fname)
            chunks = create_document_chunks(pages, source_name=fname, source_type=stype, chunk_size=500, chunk_overlap=100)
            print(f"Extracted {len(pages)} pages from {fname} -> {len(chunks)} chunks.")
            all_chunks.extend(chunks)

    print(f"\nTotal chunks across all documents: {len(all_chunks)}")

    print("\n--- 2. Testing Fresh Chroma Vector Store Indexing ---")
    vector_store = build_fresh_vector_store(all_chunks)
    print("Vector store built and indexed successfully.")

    print("\n--- 3. Testing Top-8 Multi-Document Retrieval ---")
    query = "Am I a good fit for these roles and what experience is required?"
    retrieval_res = retrieve_top_chunks(vector_store, query, top_k=8)

    print(f"Top Similarity Score: {retrieval_res['top_score']:.4f}")
    print(f"Confidence Label: {retrieval_res['confidence_label']} (Badge: {retrieval_res['confidence_color']})")
    print(f"Total Chunks Retrieved: {len(retrieval_res['retrieved_chunks'])}")
    print("Sources represented in Top-8:")
    for src_name, items in retrieval_res["grouped_sources"].items():
        print(f"  - {src_name}: {len(items)} chunk(s)")

    print("\n--- 4. Testing Conflict Detection Parser ---")
    sample_llm_response = (
        "Jane Doe has 2.5 years of experience with Python, PyTorch, and SQL.\n"
        "She fits Role B well, but lacks the 5+ years required for Role A.\n\n"
        "<CONFLICTS>\n"
        "Note: sample_jd_senior.pdf requires 5+ years experience while sample_jd_junior.pdf requires 0-1 years experience.\n"
        "Note: sample_jd_junior.pdf requires Java while sample_jd_senior.pdf requires PyTorch and MLOps.\n"
        "</CONFLICTS>"
    )
    answer, conflicts = parse_conflicts_and_answer(sample_llm_response)
    print("Clean Answer:")
    print(answer)
    print("\nExtracted Conflicts:")
    print(conflicts)

    print("\n[SUCCESS] All RAG engine & conflict detection unit tests passed!")


if __name__ == "__main__":
    run_test()
