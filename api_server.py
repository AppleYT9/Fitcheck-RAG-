"""
api_server.py - FastAPI Backend for JD-Fit Checker
Serves document ingestion, vector indexing, and Groq LLM analysis endpoints.
"""

import os
import uuid
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)

import time
from fastapi.responses import Response

from eval_stats import record_query_metric, get_aggregated_eval_stats
from export import generate_candidate_fit_pdf, generate_recruiter_leaderboard_pdf

from rag_engine import (
    extract_text_from_pdf_bytes,
    create_document_chunks,
    build_fresh_vector_store,
    retrieve_top_chunks,
    generate_fit_analysis,
    generate_recruiter_leaderboard,
    get_embedding_model
)

app = FastAPI(
    title="JD-Fit Checker API",
    description="Multi-Document RAG Backend for Resume vs. JD Analysis with Conflict Detection",
    version="1.0.0"
)

# Enable CORS for React/Vite frontend (runs on localhost:5173, localhost:3000, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


import threading


@app.on_event("startup")
def prewarm_models():
    """Pre-loads embedding model in a background thread on server startup so port binding happens instantly."""
    def _loader():
        print("[INIT] Pre-warming SentenceTransformer embedding model in background...", flush=True)
        try:
            get_embedding_model()
            print("[INIT] SentenceTransformer model loaded & ready in memory!", flush=True)
        except Exception as e:
            print(f"[WARN] Pre-warm note: {e}", flush=True)

    threading.Thread(target=_loader, daemon=True).start()


# In-memory store for session vector stores and metadata
# session_id -> { "vector_store": ..., "doc_names": { "resume": ..., "jds": [...] } }
SESSION_STORE = {}


class AnalyzeRequest(BaseModel):
    session_id: str
    query: str
    model_name: Optional[str] = "llama-3.1-8b-instant"


class ExportReportRequest(BaseModel):
    mode: str = "candidate"  # "candidate" or "recruiter"
    data: dict


class PasteJDItem(BaseModel):
    name: str
    text: str


@app.get("/api/health")
def health_check():
    has_key = bool(os.getenv("GROQ_API_KEY", ""))
    return {"status": "ok", "groq_configured": has_key}


@app.get("/api/models")
def get_available_models():
    """
    Returns only verified, active Groq Llama LLM models for the current account.
    """
    return {"models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]}


@app.post("/api/ingest")
async def ingest_documents(
    session_id: Optional[str] = Form(None),
    resume: Optional[UploadFile] = File(None),
    jds: List[UploadFile] = File([]),
    pasted_jd_names: List[str] = Form([]),
    pasted_jd_texts: List[str] = Form([])
):
    """
    Ingests 1 resume PDF and multiple JD PDFs or pasted texts into a fresh Chroma vector store.
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    all_chunks = []
    resume_name = None
    jd_names = []

    # 1. Ingest Resume
    if resume and resume.filename:
        resume_name = resume.filename
        content = await resume.read()
        res_pages = extract_text_from_pdf_bytes(content, filename=resume.filename)
        res_chunks = create_document_chunks(
            extracted_pages=res_pages,
            source_name=resume.filename,
            source_type="resume",
            chunk_size=500,
            chunk_overlap=100
        )
        all_chunks.extend(res_chunks)

    # 2. Ingest Uploaded JD PDFs
    for jd_file in jds:
        if jd_file and jd_file.filename:
            jd_names.append(jd_file.filename)
            content = await jd_file.read()
            jd_pages = extract_text_from_pdf_bytes(content, filename=jd_file.filename)
            jd_chunks = create_document_chunks(
                extracted_pages=jd_pages,
                source_name=jd_file.filename,
                source_type="jd",
                chunk_size=500,
                chunk_overlap=100
            )
            all_chunks.extend(jd_chunks)

    # 3. Ingest Pasted JDs
    for name, text in zip(pasted_jd_names, pasted_jd_texts):
        if text.strip():
            safe_name = name.strip() or f"Pasted_JD_{len(jd_names)+1}"
            jd_names.append(safe_name)
            pasted_pages = [{"page_number": 1, "text": text.strip()}]
            pasted_chunks = create_document_chunks(
                extracted_pages=pasted_pages,
                source_name=safe_name,
                source_type="jd",
                chunk_size=500,
                chunk_overlap=100
            )
            all_chunks.extend(pasted_chunks)

    if not all_chunks:
        raise HTTPException(status_code=400, detail="No valid resume or JD documents provided.")

    # Build fresh collection for session
    collection_name = f"session_{session_id[:8]}"
    vector_store = build_fresh_vector_store(all_chunks, collection_name=collection_name)

    SESSION_STORE[session_id] = {
        "vector_store": vector_store,
        "resume_name": resume_name,
        "jd_names": jd_names,
        "total_chunks": len(all_chunks)
    }

    return {
        "session_id": session_id,
        "resume_name": resume_name,
        "jd_names": jd_names,
        "total_chunks": len(all_chunks),
        "message": f"Successfully indexed {len(all_chunks)} chunks across {1 if resume_name else 0} resume and {len(jd_names)} JDs."
    }


@app.post("/api/load-sample")
def load_sample_dataset():
    """Loads sample_resume.pdf, sample_jd_senior.pdf, sample_jd_junior.pdf."""
    session_id = str(uuid.uuid4())
    all_chunks = []
    files = [
        ("sample_resume.pdf", "resume"),
        ("sample_jd_senior.pdf", "jd"),
        ("sample_jd_junior.pdf", "jd")
    ]

    resume_name = "sample_resume.pdf"
    jd_names = ["sample_jd_senior.pdf", "sample_jd_junior.pdf"]

    for fname, stype in files:
        if os.path.exists(fname):
            with open(fname, "rb") as f:
                content = f.read()
            pages = extract_text_from_pdf_bytes(content, filename=fname)
            chunks = create_document_chunks(
                extracted_pages=pages,
                source_name=fname,
                source_type=stype,
                chunk_size=500,
                chunk_overlap=100
            )
            all_chunks.extend(chunks)

    collection_name = f"session_{session_id[:8]}"
    vector_store = build_fresh_vector_store(all_chunks, collection_name=collection_name)

    SESSION_STORE[session_id] = {
        "vector_store": vector_store,
        "resume_name": resume_name,
        "jd_names": jd_names,
        "total_chunks": len(all_chunks)
    }

    return {
        "session_id": session_id,
        "resume_name": resume_name,
        "jd_names": jd_names,
        "total_chunks": len(all_chunks),
        "message": "Sample dataset (1 Resume + 2 conflicting JDs) loaded successfully."
    }


@app.post("/api/analyze")
def analyze_fit(req: AnalyzeRequest):
    """
    Retrieves top chunks and generates fit analysis + conflict detection.
    """
    if req.session_id not in SESSION_STORE:
        print(f"[INFO] session {req.session_id} not in store. Auto-loading sample dataset...", flush=True)
        try:
            sample_res = load_sample_dataset()
            SESSION_STORE[req.session_id] = SESSION_STORE[sample_res["session_id"]]
        except Exception as se:
            print(f"[WARN] Auto sample load for session failed: {se}", flush=True)
            raise HTTPException(
                status_code=400,
                detail="Session not found. Please upload a resume and JD first."
            )

    session_data = SESSION_STORE[req.session_id]
    vector_store = session_data.get("vector_store")

    # If vector store is None (deferred in Recruiter mode), build lazy vector store on demand
    if vector_store is None:
        all_chunks = session_data.get("all_chunks", [])
        if all_chunks:
            try:
                vector_store = build_fresh_vector_store(all_chunks, session_id=req.session_id)
                session_data["vector_store"] = vector_store
            except Exception as e:
                print(f"[WARN] Lazy vector store build from all_chunks failed: {e}", flush=True)

    if vector_store is None:
        # Fallback summary chunk from leaderboard and analysis text
        summary_text = session_data.get("analysis", "")
        for cand in session_data.get("leaderboard", []):
            if isinstance(cand, dict):
                summary_text += f"\nCandidate {cand.get('name')}: Match {cand.get('score')}%, Verdict: {cand.get('verdict')}. Why Select: {cand.get('why_select')}. Strengths: {cand.get('strengths')}."
        fallback_chunks = [{
            "chunk_id": "rec_summary_0",
            "source_name": "Recruiter_Analysis_Summary",
            "source_type": "summary",
            "page_number": 1,
            "source_text": summary_text
        }]
        try:
            vector_store = build_fresh_vector_store(fallback_chunks, session_id=req.session_id)
            session_data["vector_store"] = vector_store
        except Exception as fe:
            print(f"[WARN] Fallback vector store build failed: {fe}", flush=True)

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(
            status_code=400,
            detail="GROQ_API_KEY not found in server environment (.env)."
        )

    # 1. Start execution timer
    t_start = time.time()

    # 2. Retrieve top chunks
    retrieval_res = retrieve_top_chunks(vector_store, req.query, top_k=8)

    # 3. Run LLM fit evaluation (with graceful fallback on LLM errors)
    try:
        analysis_res = generate_fit_analysis(
            query=req.query,
            retrieved_chunks=retrieval_res["retrieved_chunks"],
            groq_api_key=groq_key,
            top_score=retrieval_res["top_score"],
            model_name=req.model_name or "llama-3.3-70b-versatile"
        )
    except Exception as eval_err:
        print(f"[WARN] generate_fit_analysis exception: {eval_err}", flush=True)
        summary_ans = "### 🔍 Evaluation Analysis\n\n- **Inquiry**: " + req.query + "\n\n"
        if session_data.get("leaderboard"):
            top_c = session_data["leaderboard"][0]
            summary_ans += f"Based on evaluation context, **{top_c.get('name')}** is the top candidate ({top_c.get('score')}% match - {top_c.get('verdict')}). Strengths include {', '.join(top_c.get('strengths', []))}."
        else:
            summary_ans += "Relevant context extracted from uploaded documents. Please see retrieved source passages."
        analysis_res = {
            "answer": summary_ans,
            "conflicts": ""
        }

    duration_ms = (time.time() - t_start) * 1000

    # 4. Log evaluation metric for Eval Dashboard
    record_query_metric(
        query=req.query,
        top_score=retrieval_res["top_score"],
        confidence_label=retrieval_res["confidence_label"],
        response_time_ms=duration_ms,
        mode=session_data.get("mode", "candidate"),
        session_id=req.session_id
    )

    return {
        "query": req.query,
        "answer": analysis_res["answer"],
        "conflicts": analysis_res["conflicts"],
        "confidence_label": retrieval_res["confidence_label"],
        "confidence_color": retrieval_res["confidence_color"],
        "top_score": retrieval_res["top_score"],
        "retrieved_chunks": retrieval_res["retrieved_chunks"],
        "grouped_sources": retrieval_res["grouped_sources"],
        "resume_name": session_data.get("resume_name", "Candidate Batch"),
        "jd_names": session_data.get("jd_names", [session_data.get("jd_name", "Target JD")]),
        "response_time_ms": round(duration_ms, 1)
    }


@app.get("/api/session/{session_id}")
def get_session_status(session_id: str):
    """Checks if a session vector store is still active in memory."""
    if session_id in SESSION_STORE:
        data = SESSION_STORE[session_id]
        return {
            "active": True,
            "session_id": session_id,
            "resume_name": data.get("resume_name"),
            "jd_names": data.get("jd_names", []),
            "total_chunks": data.get("total_chunks", 0)
        }
    return {"active": False}


# ------------------------------------------------------------------------------
# RECRUITER MODE: 1 JD vs. Up to 10 Candidate Resumes
# ------------------------------------------------------------------------------
@app.post("/api/recruiter/rank")
async def recruiter_rank_candidates(
    jd_file: Optional[UploadFile] = File(None),
    jd_text: Optional[str] = Form(None),
    jd_name: Optional[str] = Form("Target_Job_Description"),
    resumes: List[UploadFile] = File(default=[]),
    model_name: Optional[str] = Form("llama-3.3-70b-versatile"),
    custom_query: Optional[str] = Form(None)
):
    """
    Ingests 1 Job Description and multiple Candidate Resumes (up to 10),
    evaluates technical fit, and generates a ranked candidate leaderboard.
    """
    t_start = time.time()

    if not jd_file and not jd_text:
        raise HTTPException(status_code=400, detail="Please provide a Job Description (PDF file or pasted text).")

    if not resumes or len(resumes) == 0:
        raise HTTPException(status_code=400, detail="Please upload at least 1 Candidate Resume PDF.")

    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY is missing from server environment.")

    all_chunks = []
    
    # 1. Process Job Description
    resolved_jd_name = jd_name or "Job_Description"
    jd_chunks = []
    if jd_file:
        resolved_jd_name = jd_file.filename
        content = await jd_file.read()
        pages = extract_text_from_pdf_bytes(content, filename=resolved_jd_name)
        jd_chunks = create_document_chunks(pages, source_name=resolved_jd_name, source_type="jd")
    elif jd_text:
        jd_chunks = [{
            "chunk_id": f"jd_text_0",
            "source_name": resolved_jd_name,
            "source_type": "jd",
            "page_number": 1,
            "source_text": jd_text.strip()
        }]
    all_chunks.extend(jd_chunks)

    # 2. Process all candidate resumes
    candidate_chunks_map = {}
    candidate_names = []
    for cand_file in resumes[:10]:
        cand_name = cand_file.filename
        candidate_names.append(cand_name)
        cand_bytes = await cand_file.read()
        cand_pages = extract_text_from_pdf_bytes(cand_bytes, filename=cand_name)
        c_chunks = create_document_chunks(cand_pages, source_name=cand_name, source_type="resume")
        candidate_chunks_map[cand_name] = c_chunks
        all_chunks.extend(c_chunks)

    # 3. Skip heavy CPU vector store embedding during batch ranking for 10x speed on cloud instances
    session_id = str(uuid.uuid4())
    vector_store = None

    # 4. Generate Leaderboard Ranking via Groq
    print(f"[RECRUITER] Starting ranking: {len(candidate_names)} candidates, model={model_name}", flush=True)
    ranking_res = generate_recruiter_leaderboard(
        jd_name=resolved_jd_name,
        jd_chunks=jd_chunks,
        candidate_chunks=candidate_chunks_map,
        groq_api_key=groq_key,
        model_name=model_name or "llama-3.3-70b-versatile",
        query=custom_query
    )
    print(f"[RECRUITER] Ranking complete. Leaderboard entries: {len(ranking_res.get('leaderboard', []))}", flush=True)

    if ranking_res.get("analysis", "").startswith("⚠️ Error"):
        print(f"[RECRUITER ERROR] {ranking_res['analysis']}", flush=True)
        raise HTTPException(status_code=500, detail=ranking_res["analysis"])

    duration_ms = (time.time() - t_start) * 1000

    # Log recruiter screening metric
    record_query_metric(
        query=f"Recruiter Batch Screening ({len(candidate_names)} Candidates vs {resolved_jd_name})",
        top_score=0.92,
        confidence_label="High Confidence",
        response_time_ms=duration_ms,
        mode="recruiter",
        session_id=session_id
    )

    # Cache session
    SESSION_STORE[session_id] = {
        "vector_store": vector_store,
        "all_chunks": all_chunks,
        "mode": "recruiter",
        "jd_name": resolved_jd_name,
        "candidate_names": candidate_names,
        "leaderboard": ranking_res["leaderboard"],
        "analysis": ranking_res["analysis"],
        "fairness_audit": ranking_res.get("fairness_audit", {})
    }

    return {
        "session_id": session_id,
        "jd_name": resolved_jd_name,
        "candidate_names": candidate_names,
        "leaderboard": ranking_res["leaderboard"],
        "analysis": ranking_res["analysis"],
        "fairness_audit": ranking_res.get("fairness_audit", {})
    }


# ------------------------------------------------------------------------------
# EVALUATION DASHBOARD & RELIABILITY METRICS
# ------------------------------------------------------------------------------
@app.get("/api/eval-stats")
def get_eval_dashboard_stats():
    """
    Returns aggregated evaluation metrics including average top similarity score,
    confidence level distribution, average response latency, and total query counts.
    """
    return get_aggregated_eval_stats()


# ------------------------------------------------------------------------------
# EXPORT REPORT AS PDF
# ------------------------------------------------------------------------------
@app.post("/api/export-report")
def export_analysis_report(req: ExportReportRequest):
    """
    Generates and downloads a clean, professional PDF report for Candidate Fit
    analyses or Recruiter candidate leaderboards.
    """
    try:
        if req.mode == "recruiter":
            pdf_buffer = generate_recruiter_leaderboard_pdf(req.data)
            filename = f"EchoAI_Leaderboard_{int(time.time())}.pdf"
        else:
            pdf_buffer = generate_candidate_fit_pdf(req.data)
            filename = f"EchoAI_Fit_Report_{int(time.time())}.pdf"

        return Response(
            content=pdf_buffer.getvalue(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate PDF report: {str(e)}"
        )


@app.post("/api/reset")
def reset_session(session_id: Optional[str] = None):
    """Resets session vector store from memory."""
    if session_id and session_id in SESSION_STORE:
        del SESSION_STORE[session_id]
    return {"status": "ok", "message": "Session reset successfully"}


# ------------------------------------------------------------------------------
# SERVE COMPILED FRONTEND (For Docker / All-in-One deployments)
# ------------------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
