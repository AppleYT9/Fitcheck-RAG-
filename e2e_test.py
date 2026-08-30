"""
e2e_test.py - Comprehensive End-to-End Test Suite for Fitcheck
Tests: ingestion, candidate mode, recruiter mode, fairness audit, eval stats, export, error handling
"""

import os
import sys
import time
import json
import io
import requests

# Fix Windows console Unicode encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

API_BASE = "http://127.0.0.1:8000"

PASS = 0
FAIL = 0
BUGS_FOUND = []
PERF_METRICS = {}

def log_result(test_name, passed, detail=""):
    global PASS, FAIL
    if passed:
        PASS += 1
        print(f"  ✅ PASS: {test_name}")
    else:
        FAIL += 1
        print(f"  ❌ FAIL: {test_name} — {detail}")
        BUGS_FOUND.append({"test": test_name, "detail": detail})
    if detail and passed:
        print(f"         ↳ {detail}")


def section(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


# ====== HELPERS ======
def create_pdf_bytes(text_content, title="Test Document"):
    """Create a minimal valid PDF with text content using reportlab."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        elements = [Paragraph(text_content.replace("\n", "<br/>"), styles['Normal'])]
        doc.build(elements)
        buf.seek(0)
        return buf.read()
    except ImportError:
        # Fallback: create a truly minimal PDF manually
        content = f"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj
4 0 obj<</Length {len(text_content) + 30}>>
stream
BT /F1 12 Tf 72 720 Td ({text_content}) Tj ET
endstream
endobj
5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj
xref
0 6
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000266 00000 n 
trailer<</Size 6/Root 1 0 R>>
startxref
{350 + len(text_content)}
%%EOF"""
        return content.encode('latin-1')


def make_resume_text(name="John Doe", skills="Python, Machine Learning, TensorFlow, NLP", experience="3 years"):
    return f"""
RESUME - {name}

PROFESSIONAL SUMMARY:
Experienced Software Engineer with {experience} of hands-on experience in {skills}.
Strong background in building production-grade ML pipelines and REST APIs.

SKILLS:
- Programming: {skills}
- Frameworks: FastAPI, Flask, Django, React
- Cloud: AWS (S3, Lambda, EC2), Docker, Kubernetes
- Databases: PostgreSQL, MongoDB, ChromaDB, Redis

EXPERIENCE:
Senior ML Engineer | TechCorp Inc. | 2021-Present
- Designed and deployed NLP classification models serving 1M+ daily requests
- Built RAG pipelines using LangChain, ChromaDB, and OpenAI embeddings
- Reduced inference latency by 40% through model quantization and caching

Software Developer | StartupXYZ | 2019-2021
- Developed REST APIs with FastAPI processing 500K requests/day
- Implemented CI/CD pipelines with GitHub Actions and Docker

EDUCATION:
B.Tech Computer Science, IIT Delhi, 2019
GPA: 8.9/10

PROJECTS:
- Resume Screener: Built an AI-powered resume screening tool using RAG and vector search
- Sentiment Analyzer: Real-time Twitter sentiment analysis using BERT fine-tuning
"""


def make_jd_text(title="Senior ML Engineer", requirements="Python, TensorFlow, 5+ years", level="Senior"):
    return f"""
JOB DESCRIPTION: {title}

ABOUT THE ROLE:
We are looking for a {level} level engineer to join our AI/ML team.

REQUIREMENTS:
- {requirements}
- Strong problem-solving and communication skills
- Experience with cloud platforms (AWS/GCP/Azure)
- Familiarity with CI/CD, Docker, and Kubernetes
- Bachelor's degree in Computer Science or related field

RESPONSIBILITIES:
- Design, develop, and deploy ML models in production
- Collaborate with cross-functional teams
- Mentor junior engineers
- Contribute to system design and architecture decisions

NICE TO HAVE:
- Publications in top ML conferences
- Open source contributions
- Experience with LLMs and RAG systems
"""


# ======================================================================
# SECTION 1: HEALTH CHECK
# ======================================================================
section("0. Health Check")
try:
    r = requests.get(f"{API_BASE}/api/health", timeout=5)
    data = r.json()
    log_result("Backend is alive", r.status_code == 200, f"Status: {data}")
    log_result("Groq API key configured", data.get("groq_configured", False) == True,
               "GROQ_API_KEY missing from .env" if not data.get("groq_configured") else "")
except Exception as e:
    log_result("Backend is alive", False, f"Cannot connect: {e}")
    print("\n⛔ Backend is not running. Start with: python run_api.py")
    sys.exit(1)


# ======================================================================
# SECTION 1: DOCUMENT INGESTION
# ======================================================================
section("1. Document Ingestion")

# 1a. Valid resume PDF
print("\n  [1a] Valid resume PDF ingestion...")
t0 = time.time()
resume_bytes = create_pdf_bytes(make_resume_text())
files = {
    "resume": ("test_resume.pdf", io.BytesIO(resume_bytes), "application/pdf"),
}
jd_bytes = create_pdf_bytes(make_jd_text())
jd_files = [("jds", ("test_jd.pdf", io.BytesIO(jd_bytes), "application/pdf"))]

try:
    r = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("test_resume.pdf", io.BytesIO(resume_bytes), "application/pdf"))] + jd_files,
        data={"session_id": "test-ingest-1"},
        timeout=30
    )
    ingest_time = (time.time() - t0) * 1000
    PERF_METRICS["ingest_single_resume_ms"] = round(ingest_time, 1)
    data = r.json()
    log_result("Ingestion returns 200", r.status_code == 200, f"Status: {r.status_code}")
    log_result("Chunk count is reasonable (>0)", data.get("total_chunks", 0) > 0,
               f"Chunks: {data.get('total_chunks', 0)}")
    log_result("Resume name preserved", data.get("resume_name") == "test_resume.pdf",
               f"Got: {data.get('resume_name')}")
    log_result("JD name preserved", "test_jd.pdf" in data.get("jd_names", []),
               f"Got: {data.get('jd_names')}")
    print(f"         ↳ Ingest time: {ingest_time:.0f}ms, Chunks: {data.get('total_chunks')}")
except Exception as e:
    log_result("Valid resume ingestion", False, str(e))

# 1b. Empty / no-text PDF (simulated)
print("\n  [1b] Empty/scanned PDF (no extractable text)...")
try:
    # Create a truly minimal PDF with no text stream
    empty_pdf = b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R>>endobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \n0000000115 00000 n \ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n178\n%%EOF"
    r = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("scanned.pdf", io.BytesIO(empty_pdf), "application/pdf"))],
        data={"session_id": "test-ingest-empty"},
        timeout=15
    )
    if r.status_code >= 400:
        log_result("Empty PDF returns error (not crash)", True, f"Status {r.status_code}: {r.text[:100]}")
    else:
        data = r.json()
        chunks = data.get("total_chunks", 0)
        log_result("Empty PDF returns error or zero chunks", chunks == 0,
                   f"Got {chunks} chunks — should be 0 or error")
except Exception as e:
    log_result("Empty PDF error handling", False, str(e))

# 1c. Corrupted / non-PDF file
print("\n  [1c] Non-PDF file renamed as .pdf...")
try:
    fake_pdf = b"This is not a PDF file at all, just plain text."
    r = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("fake.pdf", io.BytesIO(fake_pdf), "application/pdf"))],
        data={"session_id": "test-ingest-corrupt"},
        timeout=15
    )
    # Should either return 4xx/5xx or handle gracefully
    log_result("Corrupted file handled gracefully (no crash)",
               r.status_code != 200 or "error" in r.text.lower(),
               f"Status: {r.status_code}, Body: {r.text[:150]}")
except Exception as e:
    log_result("Corrupted file error handling", False, str(e))

# 1d. Duplicate document upload
print("\n  [1d] Duplicate document upload in same session...")
try:
    r1 = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("resume_dup.pdf", io.BytesIO(resume_bytes), "application/pdf")),
               ("jds", ("jd_dup.pdf", io.BytesIO(jd_bytes), "application/pdf"))],
        data={"session_id": "test-ingest-dup1"},
        timeout=15
    )
    chunks1 = r1.json().get("total_chunks", 0)
    
    # Upload same docs again with SAME session_id
    r2 = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("resume_dup.pdf", io.BytesIO(resume_bytes), "application/pdf")),
               ("jds", ("jd_dup.pdf", io.BytesIO(jd_bytes), "application/pdf"))],
        data={"session_id": "test-ingest-dup1"},
        timeout=15
    )
    chunks2 = r2.json().get("total_chunks", 0)
    # New ingestion replaces old session, so chunks should be same count, not doubled
    log_result("No duplicate chunks on re-upload",
               chunks2 == chunks1,
               f"First: {chunks1} chunks, Second: {chunks2} chunks")
except Exception as e:
    log_result("Duplicate upload handling", False, str(e))


# ======================================================================
# SECTION 2: CANDIDATE MODE
# ======================================================================
section("2. Candidate Mode")

# Setup: ingest sample dataset
print("\n  [Setup] Loading sample dataset...")
try:
    r = requests.post(f"{API_BASE}/api/load-sample", timeout=30)
    sample_data = r.json()
    sample_session = sample_data["session_id"]
    log_result("Sample dataset loaded", r.status_code == 200,
               f"Session: {sample_session}, Chunks: {sample_data.get('total_chunks')}")
except Exception as e:
    log_result("Sample dataset load", False, str(e))
    sample_session = None

if sample_session:
    # 2a. Basic fit question
    print("\n  [2a] Basic fit question (grounded answer)...")
    t0 = time.time()
    try:
        r = requests.post(f"{API_BASE}/api/analyze", json={
            "session_id": sample_session,
            "query": "How well does this candidate's skills match the job requirements?",
            "model_name": "openai/gpt-oss-20b"
        }, timeout=60)
        candidate_query_time = (time.time() - t0) * 1000
        PERF_METRICS["candidate_query_ms"] = round(candidate_query_time, 1)
        data = r.json()
        log_result("Analyze returns 200", r.status_code == 200)
        log_result("Answer is non-empty", len(data.get("answer", "")) > 50,
                   f"Answer length: {len(data.get('answer', ''))}")
        log_result("Confidence label present", data.get("confidence_label", "") != "",
                   f"Confidence: {data.get('confidence_label')}")
        log_result("Top score is numeric", isinstance(data.get("top_score"), (int, float)),
                   f"Score: {data.get('top_score')}")
        log_result("Retrieved chunks returned", len(data.get("retrieved_chunks", [])) > 0,
                   f"Chunks: {len(data.get('retrieved_chunks', []))}")
        print(f"         ↳ Query time: {candidate_query_time:.0f}ms")
        print(f"         ↳ Confidence: {data.get('confidence_label')} (score: {data.get('top_score'):.3f})")
    except Exception as e:
        log_result("Basic fit question", False, str(e))

    # 2b. Conflict detection with deliberately conflicting JDs
    print("\n  [2b] Conflict detection (conflicting JDs)...")
    try:
        jd_java = create_pdf_bytes(make_jd_text(
            title="Java Backend Lead",
            requirements="Java, Spring Boot, 5+ years enterprise experience, on-site only",
            level="Senior"
        ))
        jd_python_fresher = create_pdf_bytes(make_jd_text(
            title="Python ML Intern",
            requirements="Python only (no Java), 0-1 years experience, remote work, fresher preferred",
            level="Junior/Intern"
        ))
        r_ingest = requests.post(f"{API_BASE}/api/ingest",
            files=[
                ("resume", ("conflict_resume.pdf", io.BytesIO(resume_bytes), "application/pdf")),
                ("jds", ("JD_Java_Senior.pdf", io.BytesIO(jd_java), "application/pdf")),
                ("jds", ("JD_Python_Fresher.pdf", io.BytesIO(jd_python_fresher), "application/pdf")),
            ],
            data={"session_id": "test-conflict-detect"},
            timeout=20
        )
        conflict_session = r_ingest.json().get("session_id", "test-conflict-detect")

        r = requests.post(f"{API_BASE}/api/analyze", json={
            "session_id": conflict_session,
            "query": "Evaluate this candidate's fit across both job descriptions. Are there any contradictions between the JDs?",
            "model_name": "openai/gpt-oss-20b"
        }, timeout=60)
        data = r.json()
        has_conflicts = bool(data.get("conflicts", "").strip())
        answer_mentions_conflict = any(word in data.get("answer", "").lower() for word in ["conflict", "contradict", "discrepan", "inconsisten"])
        log_result("Conflict detection triggered",
                   has_conflicts or answer_mentions_conflict,
                   f"Conflicts field: '{data.get('conflicts', '')[:100]}', Answer mentions conflict: {answer_mentions_conflict}")
    except Exception as e:
        log_result("Conflict detection", False, str(e))

    # 2c. Irrelevant question (no hallucination)
    print("\n  [2c] Irrelevant question (hallucination guard)...")
    try:
        r = requests.post(f"{API_BASE}/api/analyze", json={
            "session_id": sample_session,
            "query": "What is the candidate's favorite pizza topping and what color is their car?",
            "model_name": "openai/gpt-oss-20b"
        }, timeout=60)
        data = r.json()
        answer = data.get("answer", "").lower()
        no_hallucination = any(phrase in answer for phrase in [
            "not found", "not mention", "no information", "cannot", "doesn't", "does not",
            "not available", "not specified", "no evidence", "not relevant", "no data",
            "unable to", "not included", "not provided", "outside", "beyond"
        ])
        log_result("No hallucination on irrelevant query", no_hallucination,
                   f"Answer preview: {answer[:200]}")
    except Exception as e:
        log_result("Hallucination guard", False, str(e))

    # 2d. Confidence badges correctness
    print("\n  [2d] Confidence badge matches similarity score...")
    try:
        r = requests.post(f"{API_BASE}/api/analyze", json={
            "session_id": sample_session,
            "query": "List the candidate's technical skills",
            "model_name": "openai/gpt-oss-20b"
        }, timeout=60)
        data = r.json()
        score = data.get("top_score", 0)
        label = data.get("confidence_label", "")
        
        expected = ""
        if score >= 0.35:
            expected = "High"
        elif score >= 0.18:
            expected = "Medium"
        else:
            expected = "Low"
        
        label_matches = expected.lower() in label.lower()
        log_result("Confidence badge matches score threshold", label_matches,
                   f"Score: {score:.3f}, Label: '{label}', Expected contains: '{expected}'")
    except Exception as e:
        log_result("Confidence badge check", False, str(e))


# ======================================================================
# SECTION 3: RECRUITER MODE
# ======================================================================
section("3. Recruiter Mode")

# 3a. Basic 3-candidate ranking
print("\n  [3a] 3-candidate ranking...")
try:
    jd_ml = create_pdf_bytes(make_jd_text("ML Engineer", "Python, TensorFlow, PyTorch, 3+ years ML"))
    res1 = create_pdf_bytes(make_resume_text("Alice Chen", "Python, TensorFlow, PyTorch, NLP, Computer Vision", "4 years"))
    res2 = create_pdf_bytes(make_resume_text("Bob Smith", "Python, scikit-learn, pandas, data analysis", "2 years"))
    res3 = create_pdf_bytes(make_resume_text("Charlie Brown", "JavaScript, React, Node.js, HTML, CSS", "3 years"))

    t0 = time.time()
    r = requests.post(f"{API_BASE}/api/recruiter/rank",
        files=[
            ("jd_file", ("ML_Engineer_JD.pdf", io.BytesIO(jd_ml), "application/pdf")),
            ("resumes", ("Alice_Chen.pdf", io.BytesIO(res1), "application/pdf")),
            ("resumes", ("Bob_Smith.pdf", io.BytesIO(res2), "application/pdf")),
            ("resumes", ("Charlie_Brown.pdf", io.BytesIO(res3), "application/pdf")),
        ],
        data={"model_name": "openai/gpt-oss-20b"},
        timeout=120
    )
    recruiter_3_time = (time.time() - t0) * 1000
    PERF_METRICS["recruiter_3_candidates_ms"] = round(recruiter_3_time, 1)

    data = r.json()
    lb = data.get("leaderboard", [])
    log_result("Recruiter rank returns 200", r.status_code == 200, f"Status: {r.status_code}")
    log_result("Leaderboard has entries", len(lb) > 0, f"Entries: {len(lb)}")
    
    if lb:
        first = lb[0]
        log_result("Leaderboard has why_select", bool(first.get("why_select", "")),
                   f"why_select: {str(first.get('why_select', ''))[:80]}")
        log_result("Leaderboard has why_not_select", bool(first.get("why_not_select", "")),
                   f"why_not_select: {str(first.get('why_not_select', ''))[:80]}")
        
        # Check that scores differentiate candidates meaningfully
        scores = [c.get("score", 0) for c in lb]
        score_range = max(scores) - min(scores) if scores else 0
        log_result("Scores differentiate candidates (range > 10pts)", score_range > 10,
                   f"Scores: {scores}, Range: {score_range}")
        
    analysis = data.get("analysis", "")
    log_result("Analysis text present", len(analysis) > 100, f"Analysis length: {len(analysis)}")
    print(f"         ↳ Recruiter 3-candidate time: {recruiter_3_time:.0f}ms")
except Exception as e:
    log_result("3-candidate ranking", False, str(e))


# ======================================================================
# SECTION 4: FAIRNESS AUDIT
# ======================================================================
section("4. Fairness Audit Panel")

# 4a. Resume WITH sensitive signals
print("\n  [4a] Resume with explicit personal signals...")
try:
    from fairness_audit import audit_resume_text
    
    resume_with_signals = """
    RESUME - Test Candidate
    Date of Birth: 15/03/1995
    Gender: Male
    Marital Status: Single
    Nationality: Indian
    Father's Name: Mr. Test Senior
    
    SKILLS: Python, Machine Learning, TensorFlow
    EXPERIENCE: 3 years at TechCorp
    """
    result = audit_resume_text(resume_with_signals)
    log_result("Detects sensitive signals", result["contains_sensitive_signals"] == True,
               f"Flagged: {result['flagged_signal_types']}")
    log_result("Detects Age/DOB", "Age / DOB" in result["flagged_signal_types"],
               f"Types: {result['flagged_signal_types']}")
    log_result("Detects Gender", "Gender Marker" in result["flagged_signal_types"],
               f"Types: {result['flagged_signal_types']}")
    log_result("Detects Marital Status", "Marital / Family Status" in result["flagged_signal_types"],
               f"Types: {result['flagged_signal_types']}")
    log_result("Detects Nationality", "Nationality / Religion / Caste" in result["flagged_signal_types"],
               f"Types: {result['flagged_signal_types']}")
except Exception as e:
    log_result("Sensitive signal detection", False, str(e))

# 4b. Resume WITHOUT sensitive signals
print("\n  [4b] Resume without personal signals (no false positives)...")
try:
    clean_resume = """
    PROFESSIONAL SUMMARY
    Experienced software engineer with 5 years of Python development.
    
    SKILLS: Python, TensorFlow, AWS, Docker, Kubernetes
    
    EXPERIENCE:
    Senior Engineer at TechCorp, 2020-2025
    - Built ML pipelines for NLP applications
    - Led a team of 4 engineers
    
    EDUCATION:
    B.Tech CS, National Institute of Technology, 2020
    """
    result = audit_resume_text(clean_resume)
    log_result("No false positives on clean resume", result["contains_sensitive_signals"] == False,
               f"Flagged types: {result['flagged_signal_types']}")
except Exception as e:
    log_result("False positive check", False, str(e))


# ======================================================================
# SECTION 5: EVAL DASHBOARD
# ======================================================================
section("5. Eval Dashboard")
print("\n  [5a] Eval stats endpoint returns valid data...")
try:
    r = requests.get(f"{API_BASE}/api/eval-stats", timeout=10)
    data = r.json()
    log_result("Eval stats returns 200", r.status_code == 200)
    log_result("Has total_queries field", "total_queries" in data, f"Keys: {list(data.keys())}")
    log_result("Has confidence_distribution", "confidence_distribution" in data)
    log_result("Has avg_top_score", "avg_top_score" in data, f"Avg score: {data.get('avg_top_score')}")
    log_result("Has recent_queries list", isinstance(data.get("recent_queries"), list),
               f"Recent queries count: {len(data.get('recent_queries', []))}")
    
    total_q = data.get("total_queries", 0)
    if total_q > 0:
        dist = data.get("confidence_distribution", {})
        high_pct = dist.get("High", {}).get("percentage", 0)
        mod_pct = dist.get("Moderate", {}).get("percentage", 0)
        low_pct = dist.get("Low", {}).get("percentage", 0)
        total_pct = high_pct + mod_pct + low_pct
        log_result("Confidence distribution sums ~100%",
                   abs(total_pct - 100.0) < 5.0,
                   f"High: {high_pct}% + Mod: {mod_pct}% + Low: {low_pct}% = {total_pct}%")
    
    print(f"         ↳ Total queries logged: {total_q}")
    print(f"         ↳ Avg similarity: {data.get('avg_top_score', 0):.3f}")
    print(f"         ↳ Avg latency: {data.get('avg_response_time_ms', 0):.0f}ms")
except Exception as e:
    log_result("Eval stats endpoint", False, str(e))


# ======================================================================
# SECTION 6: EXPORT PDF
# ======================================================================
section("6. Export Feature")

# 6a. Candidate mode PDF
print("\n  [6a] Export candidate fit report PDF...")
try:
    export_data = {
        "mode": "candidate",
        "data": {
            "query": "How well does this candidate match?",
            "resume_name": "test_resume.pdf",
            "jd_names": ["JD_Senior.pdf", "JD_Junior.pdf"],
            "answer": "## Overall Fit: 85%\n\nThe candidate shows **strong alignment** with the requirements.\n\n### Matching Strengths:\n- Python & TensorFlow expertise\n- 3+ years experience\n\n### Gaps:\n- No Kubernetes experience mentioned",
            "conflicts": "JD_Senior requires 5+ years while JD_Junior requires 0-1 years.",
            "confidence_label": "High Confidence",
            "top_score": 0.82,
            "model_name": "openai/gpt-oss-20b",
            "timestamp": "2026-08-30 16:00:00"
        }
    }
    r = requests.post(f"{API_BASE}/api/export-report", json=export_data, timeout=15)
    log_result("PDF export returns 200", r.status_code == 200, f"Status: {r.status_code}")
    log_result("Response is PDF content-type", "pdf" in r.headers.get("content-type", ""),
               f"Content-Type: {r.headers.get('content-type')}")
    log_result("PDF has valid header", r.content[:5] == b"%PDF-",
               f"First bytes: {r.content[:20]}")
    log_result("PDF size is reasonable (>1KB)", len(r.content) > 1024,
               f"Size: {len(r.content)} bytes")
except Exception as e:
    log_result("Candidate PDF export", False, str(e))

# 6b. Recruiter mode PDF
print("\n  [6b] Export recruiter leaderboard PDF...")
try:
    export_data = {
        "mode": "recruiter",
        "data": {
            "jd_name": "ML_Engineer_JD.pdf",
            "leaderboard": [
                {"rank": 1, "candidate_name": "Alice Chen", "match_score": 92,
                 "verdict": "Top Pick", "why_select": "Strong ML background with TensorFlow",
                 "why_not_select": "Minor gap in Kubernetes", "interview_strategy": "Deep dive on PyTorch architectures"},
                {"rank": 2, "candidate_name": "Bob Smith", "match_score": 71,
                 "verdict": "Shortlisted", "why_select": "Good data analysis skills",
                 "why_not_select": "Lacks deep learning experience", "interview_strategy": "Test ML fundamentals"},
            ],
            "analysis": "Alice is the clear frontrunner with direct ML experience.",
            "timestamp": "2026-08-30 16:00:00",
            "model_name": "openai/gpt-oss-20b"
        }
    }
    r = requests.post(f"{API_BASE}/api/export-report", json=export_data, timeout=15)
    log_result("Recruiter PDF export returns 200", r.status_code == 200)
    log_result("Recruiter PDF has valid header", r.content[:5] == b"%PDF-")
    log_result("Recruiter PDF size reasonable", len(r.content) > 1024,
               f"Size: {len(r.content)} bytes")
except Exception as e:
    log_result("Recruiter PDF export", False, str(e))

# 6c. Special characters in PDF
print("\n  [6c] PDF with special characters (accented, symbols)...")
try:
    export_data = {
        "mode": "candidate",
        "data": {
            "query": "Évaluation du candidat",
            "resume_name": "résumé_café.pdf",
            "jd_names": ["rôle_développeur.pdf"],
            "answer": "Le candidat possède des compétences en développement. Très bon profil — score: 89%. Symbols: ™ © ® € £ ¥",
            "conflicts": "",
            "confidence_label": "High Confidence",
            "top_score": 0.89,
            "model_name": "openai/gpt-oss-20b",
            "timestamp": "2026-08-30 16:00:00"
        }
    }
    r = requests.post(f"{API_BASE}/api/export-report", json=export_data, timeout=15)
    log_result("Special char PDF doesn't crash", r.status_code == 200,
               f"Status: {r.status_code}")
except Exception as e:
    log_result("Special char PDF", False, str(e))


# ======================================================================
# SECTION 7: SESSION & AUTH
# ======================================================================
section("7. Session & Auth")

# 7a. Session isolation
print("\n  [7a] Session isolation (different sessions don't leak data)...")
try:
    # Create two independent sessions
    r1 = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("session_a_resume.pdf", io.BytesIO(resume_bytes), "application/pdf")),
               ("jds", ("session_a_jd.pdf", io.BytesIO(jd_bytes), "application/pdf"))],
        data={"session_id": "test-isolation-a"},
        timeout=20
    )
    sess_a = r1.json().get("session_id")

    r2 = requests.post(f"{API_BASE}/api/ingest",
        files=[("resume", ("session_b_resume.pdf", io.BytesIO(resume_bytes), "application/pdf")),
               ("jds", ("session_b_jd.pdf", io.BytesIO(jd_bytes), "application/pdf"))],
        data={"session_id": "test-isolation-b"},
        timeout=20
    )
    sess_b = r2.json().get("session_id")

    # Verify session A doesn't bleed into session B
    r_check_a = requests.get(f"{API_BASE}/api/session/{sess_a}", timeout=5)
    r_check_b = requests.get(f"{API_BASE}/api/session/{sess_b}", timeout=5)
    log_result("Sessions are independently active",
               r_check_a.json().get("active") and r_check_b.json().get("active"))
    
    # Try using wrong session_id
    r_wrong = requests.post(f"{API_BASE}/api/analyze", json={
        "session_id": "nonexistent-session-id",
        "query": "test",
    }, timeout=10)
    log_result("Invalid session returns 404", r_wrong.status_code == 404,
               f"Status: {r_wrong.status_code}")
except Exception as e:
    log_result("Session isolation", False, str(e))

# 7b. Session reset
print("\n  [7b] Session reset endpoint...")
try:
    r = requests.post(f"{API_BASE}/api/reset", json={"session_id": "test-isolation-a"}, timeout=5)
    log_result("Reset returns 200", r.status_code == 200)
    r_after = requests.get(f"{API_BASE}/api/session/test-isolation-a", timeout=5)
    log_result("Session no longer active after reset", r_after.json().get("active") == False)
except Exception as e:
    log_result("Session reset", False, str(e))


# ======================================================================
# SECTION 8: ERROR HANDLING
# ======================================================================
section("8. Error Handling & Resilience")

# 8a. Missing GROQ_API_KEY behavior (we can't actually remove it, but check health endpoint)
print("\n  [8a] Health endpoint reports API key status...")
try:
    r = requests.get(f"{API_BASE}/api/health", timeout=5)
    data = r.json()
    log_result("Health endpoint works", r.status_code == 200)
    log_result("Reports groq_configured status", "groq_configured" in data,
               f"groq_configured: {data.get('groq_configured')}")
except Exception as e:
    log_result("Health endpoint", False, str(e))

# 8b. Analyze without uploading documents first
print("\n  [8b] Analyze without documents (session not found)...")
try:
    r = requests.post(f"{API_BASE}/api/analyze", json={
        "session_id": "completely-fake-session",
        "query": "test query"
    }, timeout=10)
    log_result("Returns 404 for missing session", r.status_code == 404,
               f"Status: {r.status_code}")
except Exception as e:
    log_result("Missing session handling", False, str(e))

# 8c. Empty ingest (no files)
print("\n  [8c] Empty ingest (no files at all)...")
try:
    r = requests.post(f"{API_BASE}/api/ingest",
        data={"session_id": "test-empty-ingest"},
        timeout=10
    )
    log_result("Empty ingest returns 400", r.status_code == 400,
               f"Status: {r.status_code}, Body: {r.text[:100]}")
except Exception as e:
    log_result("Empty ingest handling", False, str(e))


# ======================================================================
# SECTION 9: PERFORMANCE BASELINE
# ======================================================================
section("9. Performance Baseline")

print(f"\n  📊 Performance Metrics Collected:")
for key, val in PERF_METRICS.items():
    print(f"     • {key}: {val:.0f}ms")

# Get eval stats for overall averages
try:
    r = requests.get(f"{API_BASE}/api/eval-stats", timeout=5)
    ev = r.json()
    print(f"\n  📈 Eval Dashboard Aggregates:")
    print(f"     • Total queries tracked: {ev.get('total_queries', 0)}")
    print(f"     • Avg similarity score: {ev.get('avg_top_score', 0):.3f} ({ev.get('avg_top_score', 0)*100:.1f}%)")
    print(f"     • High confidence rate: {ev.get('high_confidence_rate', 0):.1f}%")
    print(f"     • Avg response time: {ev.get('avg_response_time_ms', 0):.0f}ms")
except Exception:
    pass


# ======================================================================
# FINAL SUMMARY
# ======================================================================
section("FINAL TEST SUMMARY")
total = PASS + FAIL
print(f"\n  ✅ Passed: {PASS}/{total}")
print(f"  ❌ Failed: {FAIL}/{total}")
print(f"  🎯 Pass Rate: {(PASS/total*100 if total else 0):.1f}%")

if BUGS_FOUND:
    print(f"\n  🐛 Bugs Found ({len(BUGS_FOUND)}):")
    for i, bug in enumerate(BUGS_FOUND, 1):
        print(f"     {i}. [{bug['test']}] {bug['detail']}")

print(f"\n  📊 Performance Baseline:")
for key, val in PERF_METRICS.items():
    human_key = key.replace("_", " ").replace("ms", "").strip()
    print(f"     • {human_key}: {val:.0f}ms")

print()
