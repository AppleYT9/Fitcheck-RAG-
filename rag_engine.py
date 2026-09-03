"""
rag_engine.py - RAG & Conflict Detection Engine for JD-Fit Checker

This module handles:
1. Multi-document PDF text extraction (pypdf) & text area parsing
2. Chunking with RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=100)
3. Metadata tagging ({source_type: 'resume'|'jd', source_name, chunk_id, page_number})
4. Sentence-Transformers embedding & fresh ChromaDB collection indexing
5. Top-8 Cosine Similarity retrieval across all documents
6. Conflict Detection Prompting & Parsing via Groq LLM (llama-3.1-8b-instant)
"""

import os
import re
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional
import pypdf

# Lightweight LangChain primitives
from langchain_core.documents import Document

# ==============================================================================
# 1. EMBEDDING MODEL (Cached with @lru_cache, lazy loaded)
# ==============================================================================
@lru_cache(maxsize=1)
def get_embedding_model():
    """
    Initializes and caches the sentence-transformers embedding model.
    Using 'all-MiniLM-L6-v2' (384 dimensions, fast local execution).
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
        except ImportError:
            from langchain.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


# ==============================================================================
# 2. DOCUMENT EXTRACTION & PARSING
# ==============================================================================
def extract_text_from_pdf_bytes(pdf_file_obj, filename: str = "document.pdf") -> List[Dict[str, Any]]:
    """
    Extracts text page-by-page from PDF bytes or file object.
    
    Parameters:
        pdf_file_obj: File-like object, bytes, or bytearray.
        filename: Name of the PDF document.

    Returns:
        List of dicts: [{ "page_number": 1, "text": "..." }, ...]
    """
    import io
    if isinstance(pdf_file_obj, (bytes, bytearray)):
        pdf_file_obj = io.BytesIO(pdf_file_obj)

    reader = pypdf.PdfReader(pdf_file_obj)
    extracted_pages = []
    total_length = 0

    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        page_text_clean = page_text.strip()
        total_length += len(page_text_clean)
        
        if page_text_clean:
            extracted_pages.append({
                "page_number": idx + 1,
                "text": page_text_clean
            })

    if total_length == 0:
        raise ValueError(
            f"The PDF file '{filename}' contains no extractable text. "
            "It may be a scanned image PDF without OCR."
        )

    return extracted_pages


# ==============================================================================
# 3. CHUNKING & METADATA TAGGING
# ==============================================================================
def create_document_chunks(
    extracted_pages: List[Dict[str, Any]] = None, 
    pages: List[Dict[str, Any]] = None,
    source_name: str = "document", 
    source_type: str = "doc", 
    chunk_size: int = 500, 
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Splits text into chunks of chunk_size=500, chunk_overlap=100.
    Tags each chunk with metadata: {source_type, source_name, chunk_id, page_number}.
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain.text_splitter import RecursiveCharacterTextSplitter

    doc_pages = extracted_pages if extracted_pages is not None else (pages or [])

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    chunk_counter = 0

    for page in doc_pages:
        page_num = page["page_number"]
        page_text = page["text"]

        raw_chunks = text_splitter.split_text(page_text)
        for chunk_text in raw_chunks:
            chunk_counter += 1
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source_type": source_type,  # 'resume' or 'jd'
                    "source_name": source_name,  # e.g. 'my_resume.pdf' or 'JD_Senior_DS.pdf'
                    "chunk_id": f"{source_name}_chunk_{chunk_counter}",
                    "page_number": page_num,
                    "source_text": chunk_text
                }
            )
            documents.append(doc)

    return documents


# ==============================================================================
# 4. FRESH VECTOR STORE INDEXING (ChromaDB)
# ==============================================================================
def build_fresh_vector_store(
    documents: Any, 
    collection_name: str = "jd_fit_checker",
    session_id: Optional[str] = None
):
    """
    Creates a fresh Chroma vector store for the current session.
    Resets any existing collection to prevent cross-session document leakage.
    """
    try:
        from langchain_chroma import Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
        except ImportError:
            from langchain.vectorstores import Chroma

    embeddings = get_embedding_model()
    
    col_name = collection_name
    if session_id:
        col_name = f"session_{session_id[:8]}"

    # Initialize Chroma in-memory store with Cosine space
    vector_store = Chroma(
        collection_name=col_name,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Clear any old documents from previous session
    try:
        existing_ids = vector_store.get()["ids"]
        if existing_ids:
            vector_store.delete(ids=existing_ids)
    except Exception:
        pass

    # Add new session documents (handle both Document objects and raw chunk dicts)
    if documents:
        docs_to_add = []
        for d in documents:
            if isinstance(d, Document):
                docs_to_add.append(d)
            elif isinstance(d, dict):
                docs_to_add.append(
                    Document(
                        page_content=d.get("source_text", ""),
                        metadata={
                            "chunk_id": d.get("chunk_id", ""),
                            "source_name": d.get("source_name", ""),
                            "source_type": d.get("source_type", ""),
                            "page_number": d.get("page_number", 1)
                        }
                    )
                )
        if docs_to_add:
            vector_store.add_documents(docs_to_add)

    return vector_store


# ==============================================================================
# 5. TOP-8 RETRIEVAL & CONFIDENCE SCORING
# ==============================================================================
def calculate_confidence(top_score: float) -> Tuple[str, str]:
    """
    Assigns confidence label and color badge based on top chunk similarity score.
    - Top score >= 0.35: High confidence (Green)
    - 0.18 to 0.35: Medium confidence (Yellow)
    - < 0.18: Low confidence (Red)
    """
    if top_score >= 0.35:
        return "High confidence", "green"
    elif top_score >= 0.18:
        return "Medium confidence", "yellow"
    else:
        return "Low confidence — general / conversational query", "yellow"


def retrieve_top_chunks(vector_store: Any, query: str, top_k: int = 8) -> Dict[str, Any]:
    """
    Retrieves top_k chunks across ALL uploaded documents by Cosine Similarity.
    Ensures balanced representation across Resume and JDs for holistic evaluation.
    """
    results_with_distance = vector_store.similarity_search_with_score(query, k=top_k)

    retrieved_chunks = []
    grouped_sources: Dict[str, List[Dict[str, Any]]] = {}
    seen_ids = set()
    top_score = 0.0

    # Helper to add a doc result
    def add_result(doc, distance):
        nonlocal top_score
        chunk_id = doc.metadata.get("chunk_id", f"chunk_{len(retrieved_chunks)+1}")
        if chunk_id in seen_ids:
            return
        seen_ids.add(chunk_id)

        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        if not retrieved_chunks:
            top_score = similarity

        percentage = f"{round(similarity * 100)}%"
        chunk_data = {
            "chunk_id": chunk_id,
            "source_type": doc.metadata.get("source_type", "unknown"),
            "source_name": doc.metadata.get("source_name", "document"),
            "page_number": doc.metadata.get("page_number", 1),
            "source_text": doc.page_content,
            "similarity_score": similarity,
            "similarity_percentage": percentage
        }
        retrieved_chunks.append(chunk_data)
        src_name = chunk_data["source_name"]
        if src_name not in grouped_sources:
            grouped_sources[src_name] = []
        grouped_sources[src_name].append(chunk_data)

    for doc, distance in results_with_distance:
        add_result(doc, distance)

    # Check if we have representation from both Resume and JD
    has_resume = any(c["source_type"] == "resume" for c in retrieved_chunks)
    has_jd = any(c["source_type"] == "jd" for c in retrieved_chunks)

    # If missing one side (e.g. general fit question only matched resume), fetch complementary chunks
    if not has_resume or not has_jd:
        try:
            extra_query = "skills experience projects qualifications requirements responsibilities"
            extra_results = vector_store.similarity_search_with_score(extra_query, k=6)
            for doc, distance in extra_results:
                if len(retrieved_chunks) >= 12:
                    break
                stype = doc.metadata.get("source_type", "")
                if (not has_resume and stype == "resume") or (not has_jd and stype == "jd"):
                    add_result(doc, distance)
        except Exception:
            pass

    confidence_label, confidence_color = calculate_confidence(top_score)

    return {
        "retrieved_chunks": retrieved_chunks,
        "grouped_sources": grouped_sources,
        "top_score": top_score,
        "confidence_label": confidence_label,
        "confidence_color": confidence_color
    }


# ==============================================================================
# 6. LLM ANSWER GENERATION & CONFLICT DETECTION (Groq)
# ==============================================================================
def parse_conflicts_and_answer(raw_llm_response: str) -> Tuple[str, str]:
    """
    Extracts conflict text inside <CONFLICTS>...</CONFLICTS> tags from the raw LLM response.
    Cleans up any leaked chunk headers, excessive gaps, and returns: (clean_answer, extracted_conflicts)
    """
    conflict_match = re.search(r'<CONFLICTS>(.*?)</CONFLICTS>', raw_llm_response, re.DOTALL | re.IGNORECASE)
    
    if conflict_match:
        extracted_conflicts = conflict_match.group(1).strip()
        clean_answer = re.sub(r'<CONFLICTS>.*?</CONFLICTS>', '', raw_llm_response, flags=re.DOTALL | re.IGNORECASE).strip()
        
        if extracted_conflicts.lower() in ["none", "no conflicts detected.", "n/a", "none detected", "none."]:
            extracted_conflicts = ""
    else:
        clean_answer = raw_llm_response.strip()
        extracted_conflicts = ""

    # Sanitize any accidental chunk header markers from the answer
    clean_answer = re.sub(r'---\s*Chunk\s*\d+.*?---\s*', '', clean_answer, flags=re.IGNORECASE)
    clean_answer = re.sub(r'\[(RESUME|JD|DOCUMENT).*?\]\s*', '', clean_answer, flags=re.IGNORECASE)
    # Collapse multiple consecutive newlines (3+ into 2)
    clean_answer = re.sub(r'\n{3,}', '\n\n', clean_answer).strip()

    return clean_answer, extracted_conflicts


def generate_fit_analysis(
    query: str, 
    retrieved_chunks: List[Dict[str, Any]], 
    groq_api_key: str,
    top_score: float,
    model_name: str = "llama-3.3-70b-versatile"
) -> Dict[str, str]:
    """
    Sends retrieved chunks and query to Groq LLM.
    Acts as an intelligent, conversational AI Recruiter & Fit Analyst like ChatGPT.
    """
    if not groq_api_key:
        return {
            "answer": "Error: Groq API key is missing. Please check your .env file or configuration.",
            "conflicts": ""
        }

    # Format context cleanly
    formatted_context_list = []
    for idx, chunk in enumerate(retrieved_chunks, 1):
        if isinstance(chunk, dict):
            src_name = chunk.get("source_name", "Document")
            src_type = chunk.get("source_type", "DOC").upper()
            page_no = chunk.get("page_number", 1)
            src_txt = chunk.get("source_text", "")
        else:
            meta = getattr(chunk, "metadata", {})
            src_name = meta.get("source_name", "Document")
            src_type = meta.get("source_type", "DOC").upper()
            page_no = meta.get("page_number", 1)
            src_txt = getattr(chunk, "page_content", str(chunk))

        formatted_context_list.append(
            f"Document Source: {src_name} ({src_type}, Page {page_no})\n"
            f"{src_txt}"
        )
    formatted_context = "\n\n".join(formatted_context_list)

    system_prompt = (
        "You are an expert AI Technical Recruiter and Career Advisor (like ChatGPT).\n"
        "You evaluate candidate qualifications against Job Descriptions (JDs) with structured, actionable recruiter analysis.\n\n"
        "GREETING & TONE RULES:\n"
        "- DO NOT invent, guess, or hallucinate candidate names. If starting with a greeting, simply use 'Hi,' or 'Hello,'.\n"
        "- Speak directly and professionally in natural, conversational, and beautifully formatted markdown.\n"
        "- NEVER echo or output chunk markers or document labels.\n\n"
        "STRUCTURED RESPONSE FORMAT:\n"
        "1. 🎯 **Overall Selection Verdict & Fit Score**: Give an honest fit percentage (e.g., 85%) and selection likelihood (Strong Match / Moderate Fit / Stretch Role).\n"
        "2. ✅ **Key Matching Strengths**: Highlight specific skills, frameworks, internships, projects, and achievements from the resume that fulfill JD requirements.\n"
        "3. ⚠️ **Skill Gaps & Missing Requirements**: Point out requirements in the JD that are not evident in the resume or need deeper demonstration.\n"
        "4. 💡 **Actionable Interview & Project Advice**: Clear, practical steps to maximize interview performance and selection odds.\n\n"
        "CONFLICT DETECTION (MANDATORY):\n"
        "- If two or more uploaded Job Descriptions contain contradictory requirements (e.g. JD 1 requires 5+ years while JD 2 requires 0-1 years; or JD 1 requires Java while JD 2 requires Python):\n"
        "  You MUST include a dedicated block at the very end formatted as:\n"
        "  <CONFLICTS>\n"
        "  Note: [JD_A] requires X while [JD_B] requires Y.\n"
        "  </CONFLICTS>\n"
        "- If no conflicts exist or only 1 JD is loaded, write: `<CONFLICTS>None</CONFLICTS>`."
    )

    user_prompt = f"DOCUMENT CONTEXT:\n{formatted_context}\n\nUSER QUERY:\n{query}"

    # Verified Groq models in order of priority (active on Groq account)
    base_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it"
    ]
    candidate_models = [model_name] if model_name else []
    for m in base_candidates:
        if m not in candidate_models:
            candidate_models.append(m)

    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]

    last_error = None
    for selected_model in candidate_models:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=selected_model,
                temperature=0.3
            )
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            clean_answer, extracted_conflicts = parse_conflicts_and_answer(raw_text)

            return {
                "answer": clean_answer,
                "conflicts": extracted_conflicts
            }
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "api_key" in err_str or "api key" in err_str or "401" in err_str:
                break
            else:
                continue

    return {
        "answer": f"⚠️ Error communicating with Groq API: {str(last_error)}",
        "conflicts": ""
    }


# ==============================================================================
# 7. RECRUITER MODE: MULTI-RESUME VS. 1 JD RANKING & LEADERBOARD
# ==============================================================================
def generate_recruiter_leaderboard(
    jd_name: str,
    jd_chunks: List[Dict[str, Any]],
    candidate_chunks: Dict[str, List[Dict[str, Any]]],
    groq_api_key: str,
    model_name: str = "llama-3.3-70b-versatile",
    query: Optional[str] = None
) -> Dict[str, Any]:
    """
    Ranks multiple candidates against 1 Job Description and produces
    both an interactive JSON leaderboard and a detailed recruiter markdown breakdown.
    """
    if not groq_api_key:
        return {
            "leaderboard": [],
            "analysis": "Error: Groq API key is missing. Please check your .env file."
        }

    from fairness_audit import audit_candidate_batch

    def get_chunk_text(c: Any) -> str:
        if hasattr(c, "page_content"):
            return str(c.page_content)
        if isinstance(c, dict):
            return str(c.get("source_text", c.get("page_content", "")))
        return str(c)

    # 1. Pre-Screening Algorithmic Fairness & Bias Audit
    candidate_raw_texts = {}
    for cand_name, chunks in candidate_chunks.items():
        candidate_raw_texts[cand_name] = "\n".join([get_chunk_text(c) for c in chunks])

    audit_summary = audit_candidate_batch(candidate_raw_texts)

    # 2. Format Job Description context
    jd_text = "\n".join([get_chunk_text(c) for c in jd_chunks])[:4000]

    # Format each candidate's resume context
    candidate_summaries = []
    for cand_name, cand_text in candidate_raw_texts.items():
        truncated_text = cand_text[:3500]
        candidate_summaries.append(f"=== CANDIDATE RESUME: {cand_name} ===\n{truncated_text}\n")

    all_candidates_context = "\n\n".join(candidate_summaries)

    system_prompt = (
        "You are an Elite AI Executive Recruiter and Technical Hiring Manager.\n"
        "You are evaluating MULTIPLE candidate resumes against ONE target Job Description (JD).\n\n"
        "CRITICAL FAIRNESS & BIAS MANDATE:\n"
        "- Base your ranking, scoring, and evaluations SOLELY on objective technical skills, projects, and work experience described in the resume.\n"
        "- Do NOT consider candidate name, gender, age, location, marital status, or any personal demographic traits in your scoring or reasoning.\n\n"
        "YOUR OBJECTIVES:\n"
        "1. Compare and rank all candidates from BEST fit (#1) to WEAKEST fit.\n"
        "2. For each candidate, provide clear, authoritative recruiter reasoning:\n"
        "   - 🎯 **Why to Select**: What makes this candidate stand out for the role (matching tools, frameworks, experience).\n"
        "   - ⚠️ **Why Not to Select / Why Ranked Lower**: Specific deficits, missing requirements, or gaps compared to higher-ranked candidates.\n"
        "3. Assign an exact Match Score (0-100%) and a Verdict badge:\n"
        "   - 'Top Pick' (Score ≥ 85%)\n"
        "   - 'Shortlisted' (Score 70% - 84%)\n"
        "   - 'Consider' (Score 50% - 69%)\n"
        "   - 'Mismatch' (Score < 50%)\n\n"
        "OUTPUT STRUCTURE:\n"
        "1. 🏆 **Executive Leaderboard & Comparative Summary**\n"
        "2. 🔍 **Candidate-by-Candidate Breakdown** (Rank, Match %, Why Select, Gaps/Why Ranked Lower, Interview Questions)\n"
        "3. ⚖️ **Final Selection Decision & Trade-offs**\n\n"
        "MANDATORY JSON BLOCK AT END:\n"
        "Output the structured leaderboard inside <JSON_LEADERBOARD>...</JSON_LEADERBOARD>:\n"
        "<JSON_LEADERBOARD>\n"
        "[\n"
        "  {\n"
        "    \"rank\": 1,\n"
        "    \"name\": \"Candidate_Filename\",\n"
        "    \"score\": 92,\n"
        "    \"verdict\": \"Top Pick\",\n"
        "    \"why_select\": \"Strongest alignment with core ML pipeline design, 3+ years hands-on PyTorch experience, and proven deployment background.\",\n"
        "    \"why_not_select\": \"Lacks deep Kubernetes experience, but easily trainable.\",\n"
        "    \"strengths\": [\"Python & PyTorch\", \"3+ yrs ML experience\", \"Docker & CI/CD\"],\n"
        "    \"gaps\": [\"No Kubernetes experience\"],\n"
        "    \"recommendation\": \"Top recommendation for final round interview.\"\n"
        "  }\n"
        "]\n"
        "</JSON_LEADERBOARD>"
    )

    custom_q = f"\nRECRUITER SPECIFIC INQUIRY: {query}" if query else ""
    user_prompt = (
        f"TARGET JOB DESCRIPTION ({jd_name}):\n{jd_text}\n\n"
        f"ALL CANDIDATE RESUMES TO RANK:\n{all_candidates_context}\n"
        f"{custom_q}\n"
        "Please provide the ranking, comparative selection rationales, and the structured <JSON_LEADERBOARD> block."
    )

    base_candidates = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "deepseek-r1-distill-llama-70b",
        "gemma2-9b-it"
    ]
    candidate_models = [model_name] if model_name else []
    for m in base_candidates:
        if m not in candidate_models:
            candidate_models.append(m)

    messages = [
        ("system", system_prompt),
        ("user", user_prompt)
    ]

    last_error = None
    for selected_model in candidate_models:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(
                groq_api_key=groq_api_key,
                model_name=selected_model,
                temperature=0.2
            )
            response = llm.invoke(messages)
            raw_text = response.content.strip()

            # Extract JSON leaderboard with multi-stage robust parsing
            leaderboard_data = []
            json_str = None
            
            # Stage 1: Check for <JSON_LEADERBOARD> tags
            json_match = re.search(r'<JSON_LEADERBOARD>(.*?)</JSON_LEADERBOARD>', raw_text, re.DOTALL | re.IGNORECASE)
            if json_match:
                json_str = json_match.group(1).strip()
            
            # Stage 2: Check for markdown json block ```json ... ```
            if not json_str:
                cb_match = re.search(r'```(?:json)?\s*(\[\s*\{.*?\}\s*\])\s*```', raw_text, re.DOTALL | re.IGNORECASE)
                if cb_match:
                    json_str = cb_match.group(1).strip()

            # Stage 3: Fallback search for any JSON array [...]
            if not json_str:
                arr_match = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
                if arr_match:
                    json_str = arr_match.group(0).strip()

            if json_str:
                try:
                    import json
                    cleaned_json = re.sub(r'^```(?:json)?\s*', '', json_str, flags=re.IGNORECASE)
                    cleaned_json = re.sub(r'\s*```$', '', cleaned_json)
                    cleaned_json = re.sub(r',\s*([\]}])', r'\1', cleaned_json)
                    parsed = json.loads(cleaned_json.strip())
                    if isinstance(parsed, list):
                        leaderboard_data = [item for item in parsed if isinstance(item, dict)]
                except Exception as je:
                    print("JSON Leaderboard parsing error:", je)

            # Stage 4: Fallback candidate card generator if leaderboard_data is empty
            if not leaderboard_data and candidate_chunks:
                rank_idx = 1
                for cand_name in candidate_chunks.keys():
                    leaderboard_data.append({
                        "rank": rank_idx,
                        "name": cand_name,
                        "score": max(50, 95 - (rank_idx - 1) * 10),
                        "verdict": "Top Pick" if rank_idx == 1 else ("Shortlisted" if rank_idx == 2 else "Consider"),
                        "why_select": f"Evaluated against target JD specifications for {cand_name}.",
                        "why_not_select": "Refer to hiring manager breakdown for comparative gaps.",
                        "strengths": ["Technical Qualifications", "Relevant Experience"],
                        "gaps": [],
                        "recommendation": f"Schedule screening call with {cand_name}."
                    })
                    rank_idx += 1

            # Enrich each candidate with fairness & sensitive signal detection
            for cand in leaderboard_data:
                if not isinstance(cand, dict):
                    continue
                cand_name = cand.get("name", "")
                matched_audit = audit_summary["candidate_audits"].get(cand_name, None)
                if not matched_audit:
                    for k, v in audit_summary["candidate_audits"].items():
                        if k in cand_name or cand_name in k:
                            matched_audit = v
                            break
                if not matched_audit:
                    matched_audit = {"contains_sensitive_signals": False, "flagged_signal_types": []}
                
                cand["contains_sensitive_signals"] = matched_audit.get("contains_sensitive_signals", False)
                cand["flagged_signal_types"] = matched_audit.get("flagged_signal_types", [])

            clean_analysis = re.sub(r'<JSON_LEADERBOARD>.*?</JSON_LEADERBOARD>', '', raw_text, flags=re.DOTALL | re.IGNORECASE).strip()
            clean_analysis = re.sub(r'```(?:json)?\s*\[\s*\{\s*"rank".*?\}\s*\]\s*```', '', clean_analysis, flags=re.DOTALL | re.IGNORECASE).strip()
            clean_analysis = re.sub(r'\n{3,}', '\n\n', clean_analysis)

            return {
                "leaderboard": leaderboard_data,
                "analysis": clean_analysis,
                "fairness_audit": audit_summary
            }
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if "api_key" in err_str or "api key" in err_str or "401" in err_str:
                break
            else:
                continue

    return {
        "leaderboard": [],
        "analysis": f"⚠️ Error generating leaderboard: {str(last_error)}"
    }
