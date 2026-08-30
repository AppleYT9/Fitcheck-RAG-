"""
export.py - Professional PDF Report Generation for Candidate Fit & Recruiter Leaderboards

PURPOSE & INTERVIEW CONTEXT:
Enterprise HR and talent operations require shareable, auditable artifacts.
This module uses ReportLab to dynamically construct structured, high-resolution PDF
documents directly from RAG analysis results, with zero external binary dependencies.
"""

import io
import re
from typing import Dict, Any, List
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY


def _clean_markdown_for_pdf(text: str) -> str:
    """Cleans raw markdown formatting and converts basic formatting to ReportLab XML tags."""
    if not text:
        return ""
    # Convert bold **text** to <b>text</b>
    cleaned = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    # Convert italic *text* to <i>text</i>
    cleaned = re.sub(r'\*(.*?)\*', r'<i>\1</i>', cleaned)
    # Replace markdown headings with clean bold headers
    cleaned = re.sub(r'^###\s+(.*)', r'<b><font size="12" color="#111827">\1</font></b>', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^##\s+(.*)', r'<b><font size="13" color="#111827">\1</font></b>', cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r'^#\s+(.*)', r'<b><font size="14" color="#111827">\1</font></b>', cleaned, flags=re.MULTILINE)
    # Bullet points
    cleaned = re.sub(r'^[-*]\s+(.*)', r'• \1', cleaned, flags=re.MULTILINE)
    # Line breaks to <br/>
    cleaned = cleaned.replace('\n', '<br/>')
    return cleaned


def generate_candidate_fit_pdf(data: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a professional Candidate Fit Analysis PDF report.
    
    Args:
        data: Dictionary containing:
            - query: Query / prompt analyzed
            - resume_name: Candidate resume name
            - jd_names: List of target job description names
            - answer: Full AI evaluation text
            - conflicts: Detected JD contradictions (optional)
            - confidence_label: 'High Confidence', etc.
            - top_score: Top retrieval score
            - model_name: AI model used
            - timestamp: Generation timestamp
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#111827')
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#4b5563')
    )

    heading2_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=colors.HexColor('#111827'),
        spaceBefore=12,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14.5,
        textColor=colors.HexColor('#1f2937')
    )

    badge_style = ParagraphStyle(
        'BadgeText',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=12,
        textColor=colors.HexColor('#15803d')
    )

    elements = []

    # 1. Header & Branding Banner
    header_table_data = [
        [
            Paragraph("<b>Fitcheck</b> • Candidate Fit Evaluation", title_style),
            Paragraph(f"<b>Date:</b> {data.get('timestamp', 'Recent')}<br/><b>Engine:</b> {data.get('model_name', 'Groq LLaMA')}", subtitle_style)
        ]
    ]
    header_table = Table(header_table_data, colWidths=[340, 190])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#111827'), spaceAfter=14))

    # 2. Metadata Box (Candidate Resume & Target JDs)
    resume_name = data.get('resume_name', 'Uploaded Resume')
    jd_names = data.get('jd_names', ['Target Job Description'])
    jds_str = ", ".join(jd_names) if isinstance(jd_names, list) else str(jd_names)
    
    conf_label = data.get('confidence_label', 'High Confidence')
    top_score = data.get('top_score', 0.0)
    score_pct = int(top_score * 100) if top_score else 85

    meta_table_data = [
        [
            Paragraph(f"<b>Candidate Resume:</b> {resume_name}", body_style),
            Paragraph(f"<b>Top Retrieval Match:</b> {score_pct}% ({conf_label})", badge_style)
        ],
        [
            Paragraph(f"<b>Evaluated Against JDs:</b> {jds_str}", body_style),
            Paragraph(f"<b>Query Focus:</b> {data.get('query', 'General Fit Evaluation')}", subtitle_style)
        ]
    ]
    meta_table = Table(meta_table_data, colWidths=[360, 170])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f9fafb')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#e5e7eb')),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(meta_table)
    elements.append(Spacer(1, 14))

    # 3. Cross-Document Conflicts Banner (if any)
    conflicts = data.get('conflicts', '')
    if conflicts and len(conflicts.strip()) > 5:
        conflict_p = Paragraph(f"<b>⚠️ Detected Cross-JD Requirement Contradictions:</b><br/>{_clean_markdown_for_pdf(conflicts)}", body_style)
        conflict_table = Table([[conflict_p]], colWidths=[530])
        conflict_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffbeb')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#fde68a')),
            ('PADDING', (0, 0), (-1, -1), 10),
        ]))
        elements.append(conflict_table)
        elements.append(Spacer(1, 14))

    # 4. Detailed AI Evaluation Breakdown
    elements.append(Paragraph("📋 Detailed Fit Analysis & Requirement Breakdown", heading2_style))
    
    answer_text = data.get('answer', 'No analysis available.')
    clean_answer = _clean_markdown_for_pdf(answer_text)
    
    # Split paragraphs for graceful page flow
    for para in clean_answer.split('<br/><br/>'):
        if para.strip():
            elements.append(Paragraph(para, body_style))
            elements.append(Spacer(1, 6))

    # 5. Footer & Authenticity Notice
    elements.append(Spacer(1, 15))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#d1d5db'), spaceAfter=8))
    footer_text = Paragraph(
        "<i>Generated by Fitcheck • Multi-Document RAG Verification Engine • ChromaDB & Groq Accelerated</i>",
        ParagraphStyle('Footer', parent=styles['Normal'], fontSize=8.5, textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER)
    )
    elements.append(footer_text)

    doc.build(elements)
    buffer.seek(0)
    return buffer


def generate_recruiter_leaderboard_pdf(data: Dict[str, Any]) -> io.BytesIO:
    """
    Generates a structured Recruiter Candidate Leaderboard & Screening PDF report.
    
    Args:
        data: Dictionary containing:
            - jd_name: Target job description name
            - leaderboard: List of ranked candidate dictionaries
            - analysis: Overall recruiter summary
            - fairness_notice: Bias audit disclaimer
            - timestamp: Generation timestamp
            - model_name: AI model used
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        'RecTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=19,
        leading=23,
        textColor=colors.HexColor('#111827')
    )

    heading2_style = ParagraphStyle(
        'RecHeading2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12.5,
        leading=16,
        textColor=colors.HexColor('#111827'),
        spaceBefore=10,
        spaceAfter=5
    )

    body_style = ParagraphStyle(
        'RecBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#1f2937')
    )

    elements = []

    # 1. Header
    jd_name = data.get('jd_name', 'Target Role')
    leaderboard = data.get('leaderboard', [])
    
    header_data = [
        [
            Paragraph(f"<b>Fitcheck</b> • Candidate Leaderboard", title_style),
            Paragraph(f"<b>Target Role:</b> {jd_name}<br/><b>Total Screened:</b> {len(leaderboard)} Candidates", ParagraphStyle('Sub', parent=styles['Normal'], fontSize=10.5, alignment=TA_RIGHT, textColor=colors.HexColor('#4b5563')))
        ]
    ]
    header_table = Table(header_data, colWidths=[330, 210])
    header_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
    elements.append(header_table)
    elements.append(Spacer(1, 8))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#111827'), spaceAfter=10))

    # 2. Fairness & Bias Audit Notice
    fairness_text = data.get('fairness_notice', (
        "<b>🛡️ Algorithmic Fairness Notice:</b> Ranking is computed strictly on technical qualifications, "
        "skills, and project experience extracted from resumes. Demographic identifiers (age, gender, nationality) "
        "were programmatically excluded from scoring."
    ))
    fairness_table = Table([[Paragraph(fairness_text, ParagraphStyle('Fair', parent=styles['Normal'], fontSize=9, leading=12.5, textColor=colors.HexColor('#065f46')))]], colWidths=[540])
    fairness_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f0fdf4')),
        ('BOX', (0, 0), (-1, -1), 0.75, colors.HexColor('#bbf7d0')),
        ('PADDING', (0, 0), (-1, -1), 7),
    ]))
    elements.append(fairness_table)
    elements.append(Spacer(1, 10))

    # 3. Overall Summary
    overall_analysis = data.get('analysis', '')
    if overall_analysis:
        elements.append(Paragraph("<b>📋 Executive Hiring Summary:</b>", heading2_style))
        elements.append(Paragraph(_clean_markdown_for_pdf(overall_analysis[:400] + ("..." if len(overall_analysis) > 400 else "")), body_style))
        elements.append(Spacer(1, 8))

    # 4. Candidate Cards
    elements.append(Paragraph("🏆 Candidate Comparative Rankings & Fit Breakdown", heading2_style))

    for idx, cand in enumerate(leaderboard):
        rank = cand.get('rank', idx + 1)
        name = cand.get('candidate_name', f"Candidate {idx + 1}")
        score = cand.get('match_score', 80)
        verdict = cand.get('verdict', 'Strong Fit')
        why_select = cand.get('why_select', cand.get('summary', 'Strong technical background.'))
        why_not = cand.get('why_not_select', 'Minor skill gaps relative to top candidate.')
        strategy = cand.get('interview_strategy', 'Assess architecture and problem solving.')

        rank_badge = f"🥇 Rank #{rank}" if rank == 1 else (f"🥈 Rank #{rank}" if rank == 2 else (f"🥉 Rank #{rank}" if rank == 3 else f"Rank #{rank}"))
        
        card_content = [
            [
                Paragraph(f"<b>{rank_badge} • {name}</b>", ParagraphStyle('CName', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=11, textColor=colors.HexColor('#111827'))),
                Paragraph(f"<b>Match Fit:</b> {score}% ({verdict})", ParagraphStyle('CScore', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=10.5, alignment=TA_RIGHT, textColor=colors.HexColor('#1d4ed8')))
            ],
            [
                Paragraph(f"<b>🎯 Why to Select:</b> {_clean_markdown_for_pdf(why_select)}", body_style),
                Paragraph("", body_style)
            ],
            [
                Paragraph(f"<b>⚠️ Deficits / Why Ranked Lower:</b> {_clean_markdown_for_pdf(why_not)}", body_style),
                Paragraph("", body_style)
            ],
            [
                Paragraph(f"<b>💡 Interview Strategy:</b> {_clean_markdown_for_pdf(strategy)}", ParagraphStyle('CStrat', parent=styles['Normal'], fontSize=9, textColor=colors.HexColor('#4b5563'))),
                Paragraph("", body_style)
            ]
        ]
        
        cand_table = Table(card_content, colWidths=[400, 140])
        cand_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb')),
            ('LINEBELOW', (0, 0), (-1, 0), 0.5, colors.HexColor('#f3f4f6')),
            ('SPAN', (0, 1), (1, 1)),
            ('SPAN', (0, 2), (1, 2)),
            ('SPAN', (0, 3), (1, 3)),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        elements.append(KeepTogether([cand_table, Spacer(1, 8)]))

    # 5. Footer
    elements.append(Spacer(1, 10))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#d1d5db'), spaceAfter=6))
    elements.append(Paragraph(
        "<i>Generated by Fitcheck • Automated Candidate Screening & Multi-Document RAG</i>",
        ParagraphStyle('FooterR', parent=styles['Normal'], fontSize=8, textColor=colors.HexColor('#9ca3af'), alignment=TA_CENTER)
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer
