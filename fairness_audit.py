"""
fairness_audit.py - Pre-Screening Bias & Fairness Audit for Automated Candidate Screening

PURPOSE & INTERVIEW CONTEXT (CRITICAL FOR INTERVIEWS):
Why does this audit exist?
Algorithmic bias is one of the most critical challenges in automated recruiting systems (e.g.,
NYC Local Law 144 compliance, EEOC guidelines). Resumes frequently contain unsolicited demographic
signals (age, gender markers, marital status, nationality, photos) that biased models might exploit.

This module acts as an algorithmic transparency layer:
1. It pre-scans resume text to detect the PRESENCE of sensitive demographic markers.
2. It explicitly discloses these signals in a Fairness Notice panel.
3. It informs the downstream LLM system prompt to STRICTLY ignore demographic markers and score
   purely on objective technical skills, projects, and work experience.

*Design Decision*: This is a DISCLOSURE/AUDIT layer rather than silent redaction, ensuring full
auditable transparency for hiring managers and recruiters.
"""

import re
from typing import Dict, Any, List, Tuple


# Regex patterns for identifying unsolicited sensitive demographic signals
SENSITIVE_SIGNAL_PATTERNS = {
    "Age / DOB": [
        r'\b(?:date\s+of\s+birth|d\.o\.b\.?|dob)\s*[:=-]?\s*\d',
        r'\b(?:born\s+in|birth\s+year)\s*[:=-]?\s*(?:19|20)\d{2}\b',
        r'\b(?:age)\s*[:=-]?\s*\d{2}\b',
        r'\b\d{2}\s*(?:years\s+old|yr\s+old|yo)\b'
    ],
    "Gender Marker": [
        r'\b(?:pronouns?)\s*[:=-]?\s*(?:he/him|she/her|they/them)',
        r'\b(?:gender|sex)\s*[:=-]?\s*(?:male|female|non-binary|transgender)\b',
        r'\b(?:mr\.|mrs\.|ms\.|miss)\s+[A-Z]'
    ],
    "Marital / Family Status": [
        r'\b(?:marital\s+status)\s*[:=-]?\s*(?:single|married|unmarried|divorced|widowed)\b',
        r'\b(?:father[\'s]*\s+name|mother[\'s]*\s+name|spouse)\s*[:=-]',
        r'\b(?:marital\s+state|family\s+status)\b'
    ],
    "Photograph Reference": [
        r'\b(?:photo\s+attached|passport\s+photo|headshot\s+enclosed|photograph\s+affixed)\b',
        r'\b(?:affix\s+(?:recent\s+)?passport\s+size\s+photo)\b'
    ],
    "Nationality / Religion / Caste": [
        r'\b(?:nationality|citizenship)\s*[:=-]?\s*[A-Za-z]+',
        r'\b(?:religion)\s*[:=-]?\s*(?:hindu|muslim|christian|sikh|jewish|buddhist|jain|agnostic|atheist)\b',
        r'\b(?:caste|category)\s*[:=-]?\s*(?:general|obc|sc|st|open)\b'
    ]
}


def audit_resume_text(resume_text: str) -> Dict[str, Any]:
    """
    Audits a single candidate resume text for the presence of sensitive demographic signals.
    
    Args:
        resume_text: Raw or extracted plain text from candidate resume.
        
    Returns:
        Dictionary containing:
        - contains_sensitive_signals: Boolean
        - flagged_signal_types: List of detected category names (e.g. ['Age / DOB', 'Gender Marker'])
        - match_count: Total count of sensitive matches found
    """
    if not resume_text:
        return {
            "contains_sensitive_signals": False,
            "flagged_signal_types": [],
            "match_count": 0
        }

    flagged_types = []
    total_matches = 0

    for category, patterns in SENSITIVE_SIGNAL_PATTERNS.items():
        found = False
        for pat in patterns:
            if re.search(pat, resume_text, re.IGNORECASE):
                found = True
                total_matches += 1
                break
        if found:
            flagged_types.append(category)

    return {
        "contains_sensitive_signals": len(flagged_types) > 0,
        "flagged_signal_types": flagged_types,
        "match_count": total_matches
    }


def audit_candidate_batch(candidate_resumes: Dict[str, str]) -> Dict[str, Any]:
    """
    Audits an entire batch of candidate resumes prior to LLM ranking.
    
    Args:
        candidate_resumes: Dict of { candidate_name: resume_text }
        
    Returns:
        Dictionary containing:
        - candidate_audits: Dict of { candidate_name: audit_dict }
        - total_candidates: Total count
        - flagged_candidates_count: Count of candidates with sensitive markers
        - fairness_notice: Formatted transparency statement
    """
    candidate_audits = {}
    flagged_names = []

    for name, text in candidate_resumes.items():
        audit_res = audit_resume_text(text)
        candidate_audits[name] = audit_res
        if audit_res["contains_sensitive_signals"]:
            flagged_names.append(name)

    flagged_count = len(flagged_names)
    total_count = len(candidate_resumes)

    if flagged_count == 0:
        notice = (
            f"This ranking is based solely on skills and experience extracted from resumes. "
            f"All {total_count} candidate resume(s) were free of explicit personal/demographic markers."
        )
    else:
        notice = (
            f"This ranking is based solely on skills and experience extracted from resumes. "
            f"{flagged_count} of {total_count} candidate(s) contained personal signals "
            f"(e.g., age, gender, marital status, nationality) which were explicitly excluded from scoring."
        )

    return {
        "candidate_audits": candidate_audits,
        "total_candidates": total_count,
        "flagged_candidates_count": flagged_count,
        "flagged_candidate_names": flagged_names,
        "fairness_notice": notice
    }
