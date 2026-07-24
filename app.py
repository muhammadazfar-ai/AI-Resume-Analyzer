
import os
import re
import json
import streamlit as st
import fitz  
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

# ==========================================
# 1. CORE LOGIC & BACKEND FUNCTIONS
# ==========================================

def extract_skills(text):
    SKILL_BANK = [
        "Python", "Java", "C++", "JavaScript", "SQL", "Git", "GitHub", 
        "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", 
        "Scikit-Learn", "Flask", "FastAPI", "Streamlit", "Docker", 
        "AWS", "Data Analysis", "NLP", "Tableau", "PowerBI"
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in SKILL_BANK:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)
            
    return {
        "skills": found_skills,
        "count": len(found_skills),
        "missing": [s for s in SKILL_BANK if s not in found_skills]
    }
# ==========================================
# AI Resume Analysis
# ==========================================

client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

def analyze_resume_with_ai(resume_text, job_description=None):

    system_prompt = (
        "You are an expert ATS recruiter. "
        "Analyze the resume and return ONLY valid JSON with these keys:\n"
        "resume_score\n"
        "ats_score\n"
        "match_percentage\n"
        "strengths\n"
        "weaknesses\n"
        "missing_keywords\n"
        "suggestions"
    )

    user_content = f"Resume:\n{resume_text}"

    if job_description:
        user_content += f"\n\nJob Description:\n{job_description}"

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_content
                }
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=2000
        )

        return json.loads(completion.choices[0].message.content)

    except Exception as e:
        return {"error": str(e)}
 

# ==========================================
# 2. STREAMLIT USER INTERFACE
# ==========================================

st.set_page_config(page_title="Ultimate AI Resume Analyzer", page_icon="📝", layout="wide")

st.title("📝 Advanced AI Resume Analyzer")
st.markdown("An end-to-end ATS parsing tool that scores resumes, detects missing skills, and matches profiles with target jobs.")

with st.sidebar:
    st.header("Project Meta")
    st.success("All Phases (1, 2, & 3) Active")
    st.info("Ensure your system terminal has initialized `GROQ_API_KEY` before running.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF resume", type=["pdf"])

with col2:
    st.subheader("2. Target Job Description (Optional - Required for Match Percentage)")
    job_desc = st.text_area("Paste the job role description details here...", height=150)

if uploaded_file is not None:
    st.success("Resume received!")
    
    with st.spinner("Parsing PDF content via PyMuPDF..."):
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            resume_text = "".join([page.get_text() for page in doc])
        except Exception as e:
            st.error(f"Could not read PDF structure: {e}")
            resume_text = ""

    if resume_text:
        tab1, tab2, tab3 = st.tabs(["📊 Skill Metrics", "🤖 AI Results Page", "📄 Raw Extracted Text"])
        
        with tab3:
            st.text_area("Plain Text Extractions", resume_text, height=400)
            
        with tab1:
            skill_results = extract_skills(resume_text)
            
            # Phase 1: Metric Blocks
            c1, c2 = st.columns(2)
            c1.metric(label="Skills Found", value=skill_results["count"])
            c2.metric(label="Predefined Skills Missing", value=len(skill_results["missing"]))
            
            if skill_results["skills"]:
                st.write("### Keywords Found in Text:")
                st.markdown(" ".join([f"`{skill}`" for skill in skill_results["skills"]]))
                
                st.write("### Keyword Distribution")
                chart_data = pd.DataFrame({
                    "Status": ["Found", "Missing"],
                    "Count": [skill_results["count"], len(skill_results["missing"])]
                })
                st.bar_chart(data=chart_data, x="Status", y="Count")
            else:
                st.warning("No default tech stack words located.")

        with tab2:
            st.write("### Smart AI Review & Alignment Dashboard")
            if st.button("Generate Full Audit Report"):
                with st.spinner("Analyzing with Llama 3 Engine via Groq..."):
                    result = analyze_resume_with_ai(resume_text, job_desc if job_desc else None)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.markdown("---")
                        
                        # --- RESULTS PAGE EXPLICIT METRICS ---
                        mc1, mc2, mc3 = st.columns(3)
                        
                        with mc1:
                            st.metric(label="Overall Resume Score", value=f"{result.get('resume_score', 0)}/100")
                            st.progress(result.get('resume_score', 0) / 100)
                            
                        with mc2:
                            st.metric(label="ATS Score", value=f"{result.get('ats_score', 0)}/100")
                            st.progress(result.get('ats_score', 0) / 100)
                            
                        with mc3:
                            if job_desc:
                                st.metric(label="Job Match Percentage", value=f"{result.get('match_percentage', 0)}%")
                                st.progress(result.get('match_percentage', 0) / 100)
                            else:
                                st.metric(label="Job Match Percentage", value="N/A")
                                st.caption("Provide a job description to activate.")

                        st.markdown("---")
                        
                        # Display structural highlights
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            st.write("### 👍 Key Strengths")
                            for strength in result.get("strengths", []):
                                st.write(f"- {strength}")
                                
                        with rc2:
                            st.write("### ⚠️ Areas for Improvement")
                            for weakness in result.get("weaknesses", []):
                                st.write(f"- {weakness}")

                        if job_desc:
                            st.write("### 🔍 Missing Keywords (Compared to Job Description)")
                            if result.get("missing_keywords"):
                                st.markdown(" ".join([f"`{kw}`" for kw in result.get("missing_keywords", [])]))
                            else:
                                st.success("No critical keywords missing from the description!")

                        st.write("### 🛠️ Step-by-Step AI Suggestions")
                        for suggestion in result.get("suggestions", []):
                            st.write(f"- {suggestion}")
                            
                        # Format text compilation for the file download feature
                        report_text = f"""RESUME ANALYSIS REPORT
======================
Resume Score: {result.get('resume_score', 0)}/100
ATS Score: {result.get('ats_score', 0)}/100
Job Match Percentage: {result.get('match_percentage', 0)}%

STRENGTHS:
{chr(10).join(['- ' + s for s in result.get('strengths', [])])}

WEAKNESSES:
{chr(10).join(['- ' + w for w in result.get('weaknesses', [])])}

MISSING KEYWORDS:
{', '.join(result.get('missing_keywords', []))}

SUGGESTIONS FOR IMPROVEMENT:
{chr(10).join(['- ' + s for s in result.get('suggestions', [])])}
"""
                        st.markdown("---")
                        st.download_button(
                            label="📥 Download Structured Report",
                            data=report_text,
                            file_name="ai_resume_analysis.txt",
                            mime="text/plain"
                        )

