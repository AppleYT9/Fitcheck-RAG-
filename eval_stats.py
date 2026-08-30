"""
eval_stats.py - Reliability & Trust Evaluation Metric Tracker for RAG Pipeline

PURPOSE & INTERVIEW CONTEXT:
In production RAG systems, tracking query performance, retrieval similarity,
and confidence distributions is critical for monitoring retrieval drift, 
hallucination risk, and latency regressions.

This module provides a lightweight, thread-safe in-memory metrics store that logs
every query execution and computes aggregated evaluation statistics.
"""

import time
import threading
from typing import List, Dict, Any

# Thread-safe lock for recording metrics across async FastAPI workers
_lock = threading.Lock()

# In-memory log of recent query metrics (capped at 500 entries to prevent memory growth)
MAX_LOG_ENTRIES = 500
_QUERY_METRICS_LOG: List[Dict[str, Any]] = []


def record_query_metric(
    query: str,
    top_score: float,
    confidence_label: str,
    response_time_ms: float,
    mode: str = "candidate",
    session_id: str = ""
) -> Dict[str, Any]:
    """
    Records a single RAG query's retrieval accuracy and latency metrics.
    
    Args:
        query: The user query string
        top_score: Highest Cosine Similarity score from retrieved chunks
        confidence_label: 'High Confidence', 'Moderate Confidence', or 'Low Confidence'
        response_time_ms: Total round-trip execution latency in milliseconds
        mode: 'candidate' or 'recruiter'
        session_id: Active session identifier
    """
    metric_entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_epoch": time.time(),
        "query": query[:120] + ("..." if len(query) > 120 else ""),
        "top_score": round(float(top_score), 4),
        "confidence_label": confidence_label,
        "response_time_ms": round(float(response_time_ms), 1),
        "mode": mode,
        "session_id": session_id
    }

    with _lock:
        _QUERY_METRICS_LOG.append(metric_entry)
        if len(_QUERY_METRICS_LOG) > MAX_LOG_ENTRIES:
            _QUERY_METRICS_LOG.pop(0)

    return metric_entry


def get_aggregated_eval_stats() -> Dict[str, Any]:
    """
    Computes aggregated system reliability and trust statistics across all recorded queries.
    
    Returns:
        Dictionary containing:
        - total_queries: Total number of queries executed
        - avg_top_score: Mean similarity score (0.0 to 1.0)
        - avg_response_time_ms: Mean latency in ms
        - confidence_distribution: Percentage and count of High / Moderate / Low queries
        - high_confidence_rate: Percentage of queries meeting high accuracy threshold
        - recent_queries: Chronological list of recent 20 query logs
    """
    with _lock:
        logs = list(_QUERY_METRICS_LOG)

    total_queries = len(logs)
    if total_queries == 0:
        return {
            "total_queries": 0,
            "avg_top_score": 0.0,
            "avg_response_time_ms": 0.0,
            "high_confidence_rate": 0.0,
            "confidence_distribution": {
                "High": {"count": 0, "percentage": 0.0},
                "Moderate": {"count": 0, "percentage": 0.0},
                "Low": {"count": 0, "percentage": 0.0}
            },
            "recent_queries": []
        }

    total_score = sum(item["top_score"] for item in logs)
    total_time = sum(item["response_time_ms"] for item in logs)

    high_count = sum(1 for item in logs if "High" in item["confidence_label"])
    mod_count = sum(1 for item in logs if "Moderate" in item["confidence_label"])
    low_count = sum(1 for item in logs if "Low" in item["confidence_label"] or "Error" in item["confidence_label"])

    avg_score = round(total_score / total_queries, 3)
    avg_time = round(total_time / total_queries, 1)
    high_rate = round((high_count / total_queries) * 100, 1)

    return {
        "total_queries": total_queries,
        "avg_top_score": avg_score,
        "avg_response_time_ms": avg_time,
        "high_confidence_rate": high_rate,
        "confidence_distribution": {
            "High": {
                "count": high_count,
                "percentage": round((high_count / total_queries) * 100, 1)
            },
            "Moderate": {
                "count": mod_count,
                "percentage": round((mod_count / total_queries) * 100, 1)
            },
            "Low": {
                "count": low_count,
                "percentage": round((low_count / total_queries) * 100, 1)
            }
        },
        "recent_queries": list(reversed(logs[-20:]))
    }
