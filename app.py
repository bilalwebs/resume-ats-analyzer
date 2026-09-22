import io
import json
import os
import re
from typing import Any

import streamlit as st
from docx import Document
from google import genai
from google.genai import types
from pypdf import PdfReader


MODEL = "gemini-2.5-flash"
MAX_FILE_SIZE_MB = 10


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Resume ATS Analyzer",
    page_icon="📄",
    layout="wide",
)


# ============================================================
# HEADER
# ============================================================

st.title("📄 Resume ATS Analyzer")

st.caption(
    "AI-powered ATS readiness analysis using Gemini Flash"
)

st.info(
    "ATS scores are estimates, not scores from a specific employer's ATS. "
    "Different ATS platforms and job descriptions use different rules."
)


# ============================================================
# API KEY
# ============================================================

def get_api_key() -> str | None:
    """
    Get Gemini API key from Streamlit Secrets first,
    then fall back to environment variables.
    """

    try:
        key = st.secrets.get("GEMINI_API_KEY")

        if key:
            return str(key).strip()

    except Exception:
        pass

    key = os.getenv("GEMINI_API_KEY")

    return key.strip() if key else None


# ============================================================
# RESUME TEXT EXTRACTION
# ============================================================

def extract_text(uploaded_file: Any) -> str:
    """
    Extract text from PDF, DOCX, or TXT files.
    """

    extension = uploaded_file.name.lower().rsplit(".", 1)[-1]

    data = uploaded_file.getvalue()

    # -------------------------
    # PDF
    # -------------------------

    if extension == "pdf":

        reader = PdfReader(io.BytesIO(data))

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    # -------------------------
    # DOCX
    # -------------------------

    if extension == "docx":

        document = Document(io.BytesIO(data))

        parts = [
            paragraph.text
            for paragraph in document.paragraphs
        ]

        # Also extract text from tables
        for table in document.tables:

            for row in table.rows:

                parts.append(
                    " | ".join(
                        cell.text
                        for cell in row.cells
                    )
                )

        return "\n".join(parts)

    # -------------------------
    # TXT
    # -------------------------

    return data.decode(
        "utf-8",
        errors="replace"
    )


# ============================================================
# JSON PARSER
# ============================================================

def parse_json(text: str) -> dict[str, Any]:
    """
    Safely parse Gemini's JSON response.
    """

    cleaned = text.strip()

    # Remove markdown JSON fences if Gemini adds them
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    match = re.search(
        r"\{.*\}",
        cleaned,
        flags=re.DOTALL,
    )

    if not match:
        raise ValueError(
            "Gemini did not return valid JSON."
        )

    result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError(
            "Gemini returned an unexpected response format."
        )

    return result


# ============================================================
# GEMINI ANALYSIS
# ============================================================

def analyze_resume(
    resume_text: str,
    job_description: str,
) -> dict[str, Any]:

    api_key = get_api_key()

    if not api_key:

        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    client = genai.Client(
        api_key=api_key
    )

    target_job = (
        job_description.strip()
        if job_description.strip()
        else
        "Not provided. Evaluate general ATS readiness."
    )

    prompt = f"""
You are a professional resume and ATS optimization analyst.

Analyze the resume against the job description when one is provided.

IMPORTANT RULES:

1. Do not invent experience.
2. Do not invent education.
3. Do not invent skills.
4. Do not invent employers.
5. Do not invent projects.
6. Do not invent metrics.
7. Do not invent achievements.
8. Recommendations must remain truthful to the resume.
9. The ATS score is an estimate, not the score of a specific ATS product.

Score the resume from 0 to 100 using these dimensions:

- ATS-readable structure
- Formatting
- Standard resume sections
- Keyword alignment
- Skills relevance
- Experience relevance
- Achievement clarity
- Measurable impact
- Date consistency
- Job-title consistency
- Overall content clarity

If a job description is not provided:

Evaluate general ATS readiness and clearly explain that
keyword matching is limited.

Return ONLY valid JSON.

Use exactly this structure:

{{
  "ats_score": 0,

  "score_label": "Strong",

  "summary": "...",

  "keyword_match": {{
    "score": 0,
    "matched_keywords": ["..."],
    "missing_keywords": ["..."],
    "notes": "..."
  }},

  "sections": [
    {{
      "section": "Experience",
      "status": "Good",
      "feedback": "..."
    }}
  ],

  "formatting": {{
    "score": 0,
    "strengths": ["..."],
    "issues": ["..."]
  }},

  "content": {{
    "score": 0,
    "strengths": ["..."],
    "issues": ["..."]
  }},

  "improvements": [
    {{
      "priority": "High",
      "area": "Professional Summary",
      "problem": "...",
      "recommendation": "..."
    }}
  ],

  "quick_wins": [
    "..."
  ],

  "ats_checklist": [
    {{
      "item": "Standard section headings",
      "status": "Pass",
      "explanation": "..."
    }}
  ]
}}

Keep all scores between 0 and 100.

Keep the improvements focused on approximately
5 to 10 genuinely useful recommendations.

JOB DESCRIPTION:

{target_job}


RESUME:

{resume_text}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
        ),
    )

    return parse_json(response.text)


# ============================================================
# SCORE VALIDATION
# ============================================================

def clamp_score(value: Any) -> int:

    try:

        return max(
            0,
            min(
                100,
                int(value)
            )
        )

    except (TypeError, ValueError):

        return 0


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload your resume",
    type=[
        "pdf",
        "docx",
        "txt",
    ],
    help=(
        "Supported formats: PDF, DOCX, TXT. "
        "Maximum size: 10 MB."
    ),
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

job_description = st.text_area(
    "Job Description (recommended)",
    height=220,
    placeholder=(
        "Paste the target job description here "
        "for targeted keyword analysis..."
    ),
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Resume",
    type="primary",
    use_container_width=True,
):

    if uploaded_file is None:

        st.warning(
            "Please upload a resume first."
        )

        st.stop()

    # File size validation

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:

        st.error(
            f"File is too large. "
            f"Please upload a file under "
            f"{MAX_FILE_SIZE_MB} MB."
        )

        st.stop()

    try:

        with st.spinner(
            "Extracting resume text and analyzing it with Gemini..."
        ):

            resume_text = extract_text(
                uploaded_file
            )

            # Basic extraction validation

            if len(resume_text.strip()) < 80:

                st.error(
                    "Very little readable text was found. "
                    "If this is a scanned PDF, upload a "
                    "text-based PDF or DOCX instead."
                )

                st.stop()

            result = analyze_resume(
                resume_text,
                job_description,
            )

        st.session_state.analysis = result

        st.session_state.resume_name = (
            uploaded_file.name
        )

    except Exception as exc:

        st.error(
            f"Analysis failed: {exc}"
        )

        st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

result = st.session_state.get(
    "analysis"
)


if result:

    st.divider()

    st.subheader(
        f"Analysis — "
        f"{st.session_state.get('resume_name', 'Resume')}"
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    score = clamp_score(
        result.get("ats_score")
    )

    keyword = result.get(
        "keyword_match",
        {}
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "ATS Readiness",
        f"{score}/100"
    )

    c2.metric(
        "Assessment",
        str(
            result.get(
                "score_label",
                "Estimated"
            )
        ),
    )

    c3.metric(
        "Keyword Match",
        f"{clamp_score(keyword.get('score'))}/100",
    )

    st.progress(
        score / 100
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    st.subheader(
        "🧾 Overall Assessment"
    )

    st.write(
        result.get(
            "summary",
            "No summary returned."
        )
    )

    # --------------------------------------------------------
    # KEYWORDS
    # --------------------------------------------------------

    st.subheader(
        "🔑 Keyword Analysis"
    )

    left, right = st.columns(2)

    with left:

        st.markdown(
            "**Matched Keywords**"
        )

        matched = keyword.get(
            "matched_keywords",
            []
        )

        st.write(
            ", ".join(
                map(str, matched)
            )
            if matched
            else
            "None identified."
        )

    with right:

        st.markdown(
            "**Missing / Useful Keywords**"
        )

        missing = keyword.get(
            "missing_keywords",
            []
        )

        st.write(
            ", ".join(
                map(str, missing)
            )
            if missing
            else
            "None identified."
        )

    if keyword.get("notes"):

        st.caption(
            keyword["notes"]
        )

    # --------------------------------------------------------
    # SECTION REVIEW
    # --------------------------------------------------------

    st.subheader(
        "📌 Section Review"
    )

    for item in result.get(
        "sections",
        []
    ):

        with st.expander(
            f"{item.get('section', 'Section')} "
            f"— {item.get('status', 'Review')}"
        ):

            st.write(
                item.get(
                    "feedback",
                    ""
                )
            )

    # --------------------------------------------------------
    # FORMATTING + CONTENT
    # --------------------------------------------------------

    st.subheader(
        "🛠️ Formatting & Content"
    )

    formatting = result.get(
        "formatting",
        {}
    )

    content = result.get(
        "content",
        {}
    )

    f1, f2 = st.columns(2)

    with f1:

        st.markdown(
            f"**Formatting: "
            f"{clamp_score(formatting.get('score'))}/100**"
        )

        for item in formatting.get(
            "strengths",
            []
        ):

            st.write(
                f"✅ {item}"
            )

        for item in formatting.get(
            "issues",
            []
        ):

            st.write(
                f"⚠️ {item}"
            )

    with f2:

        st.markdown(
            f"**Content: "
            f"{clamp_score(content.get('score'))}/100**"
        )

        for item in content.get(
            "strengths",
            []
        ):

            st.write(
                f"✅ {item}"
            )

        for item in content.get(
            "issues",
            []
        ):

            st.write(
                f"⚠️ {item}"
            )

    # --------------------------------------------------------
    # IMPROVEMENTS
    # --------------------------------------------------------

    st.subheader(
        "🚀 Recommended Improvements"
    )

    for index, item in enumerate(
        result.get(
            "improvements",
            []
        ),
        start=1,
    ):

        with st.expander(
            f"{index}. "
            f"[{item.get('priority', 'Medium')}] "
            f"{item.get('area', 'Resume')}"
        ):

            st.markdown(
                f"**Problem:** "
                f"{item.get('problem', '')}"
            )

            st.markdown(
                f"**Recommendation:** "
                f"{item.get('recommendation', '')}"
            )

    # --------------------------------------------------------
    # QUICK WINS
    # --------------------------------------------------------

    st.subheader(
        "⚡ Quick Wins"
    )

    for item in result.get(
        "quick_wins",
        []
    ):

        st.write(
            f"• {item}"
        )

    # --------------------------------------------------------
    # CHECKLIST
    # --------------------------------------------------------

    st.subheader(
        "✅ ATS Checklist"
    )

    for item in result.get(
        "ats_checklist",
        []
    ):

        status = str(
            item.get(
                "status",
                "Review"
            )
        ).lower()

        icon = (
            "✅"
            if status in {
                "pass",
                "good",
                "yes",
            }
            else
            "⚠️"
        )

        st.write(
            f"{icon} "
            f"**{item.get('item', '')}** "
            f"— "
            f"{item.get('explanation', '')}"
        )

    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

    st.caption(
        "Privacy: the resume text is sent to Google's "
        "Gemini API for analysis. Do not upload confidential "
        "documents unless you are comfortable with that processing."
    )