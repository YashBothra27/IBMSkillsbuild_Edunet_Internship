import streamlit as st
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import PyPDF2 as pdf
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from io import BytesIO
import re

# 1. PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
st.set_page_config(page_title="ResumAI", layout="wide", page_icon="🚀")

# 2. Load Environment Variables
load_dotenv()
@st.cache_resource
def get_gemini_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GOOGLE_API_KEY"]
        except (KeyError, FileNotFoundError):
            st.error("🚨 API Key missing! Set GOOGLE_API_KEY in .env or secrets.toml")
            st.stop()
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version="v1") 
    )

client = get_gemini_client()
# 3. Setup Session State (Single Initialization)
if 'history' not in st.session_state:
    st.session_state['history'] = []

if 'resume_data' not in st.session_state:
    st.session_state['resume_data'] = {
        'name': '',
        'email': '',
        'phone': '',
        'role': '',
        'linkedin': '',
        'github': '',
        'education': '',
        'skills': '',
        'experience': '',
        'projects': ''
    }

def add_to_history(action, details):
    st.session_state['history'].insert(0, f"{action} - {details}")

# 4. OPTIMIZED AI CONFIGURATION
AVAILABLE_MODELS = ["gemini-2.0-flash"]
def get_gemini_response(prompt, stream=False):

    errors = []
    for model_name in AVAILABLE_MODELS:
        try:
            if stream:
                return client.models.generate_content_stream(
                    model=model_name,
                    contents=prompt
                )
            else:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text
        except Exception as e:
            errors.append(f"{model_name} failed: {str(e)}")
            continue
    return f"All models failed. Errors: {errors}"

def clean_ai_response(text):
    """Removes Markdown code blocks (```html ... ```) from AI response"""
    text = text.replace("```html", "").replace("```", "")
    return text.strip()

# --- PDF GENERATION ---
def create_pdf(text_content):
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=LETTER,
        rightMargin=40, leftMargin=40, 
        topMargin=40, bottomMargin=40
    )
    styles = getSampleStyleSheet()
    
    # Update Normal Style
    styles['Normal'].fontSize = 9.5      
    styles['Normal'].leading = 11        
    
    # Update Headings
    styles['Heading1'].fontSize = 18
    styles['Heading1'].spaceAfter = 6    
    styles['Heading2'].fontSize = 12
    styles['Heading2'].spaceBefore = 6   
    styles['Heading2'].spaceAfter = 4    
    styles['Heading2'].textColor = "black" 
    
    # Create Heading 3
    if 'Heading3' not in styles:
        styles.add(ParagraphStyle(name='Heading3', parent=styles['Heading2']))
    styles['Heading3'].fontSize = 10.5
    styles['Heading3'].spaceBefore = 4
    styles['Heading3'].spaceAfter = 2
    
    # Create Bullet Style
    if 'Bullet' not in styles:
        styles.add(ParagraphStyle(
            name='Bullet',
            parent=styles['Normal'],
            alignment=TA_LEFT,
            leftIndent=15,
            firstLineIndent=-15, 
            spaceAfter=1,
            leading=11
        ))

    story = []
    lines = text_content.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Handle Formatting
        if line.startswith('### '):
            clean_line = line.replace('### ', '').replace('**', '')
            p = Paragraph(f"<b>{clean_line}</b>", styles['Heading3'])
            story.append(p)    
        elif line.startswith('## '):
            clean_line = line.replace('## ', '').replace('**', '')
            p = Paragraph(clean_line, styles['Heading2'])
            story.append(p)   
        elif line.startswith('# '):
            clean_line = line.replace('# ', '').replace('**', '')
            p = Paragraph(clean_line, styles['Heading1'])
            story.append(p)
            story.append(Spacer(1, 4))    
        elif line.startswith('* ') or line.startswith('- '):
            clean_line = line[2:].strip()
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', clean_line)
            p = Paragraph(f"• {formatted_line}", styles['Bullet'])
            story.append(p)    
        else:
            formatted_line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
            p = Paragraph(formatted_line, styles['Normal'])
            story.append(p)
            story.append(Spacer(1, 2))    
    doc.build(story)
    buffer.seek(0)
    return buffer

def extract_text_from_pdf(uploaded_file):
    text = ""
    try:
        reader = pdf.PdfReader(uploaded_file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
        return text
    except Exception as e:
        return str(e)

def calculate_ats_score(resume_text, job_description):
    documents = [resume_text, job_description]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    match_percentage = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return round(match_percentage * 100, 2)

# --- HTML PORTFOLIO GENERATOR ---
def generate_portfolio_html(name, role, bio, projects, email, linkedin, github):
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{name} - Portfolio</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
        :root {{
            --primary-color: #2563eb;
            --bg-color: #f8fafc;
            --text-color: #334155;
            --card-bg: #ffffff;
        }}
        body {{
            font-family: 'Roboto', sans-serif;
            margin: 0;
            padding: 0;
            background-color: var(--bg-color);
            color: var(--text-color);
            line-height: 1.6;
        }}
        a {{ text-decoration: none; color: inherit; transition: 0.3s; }}
        a:hover {{ color: var(--primary-color); }}
        
        header {{
            background: #1e293b;
            color: white;
            padding: 4rem 2rem;
            text-align: center;
        }}
        header h1 {{ margin: 0; font-size: 3.5rem; letter-spacing: 2px; }}
        header p {{ font-size: 1.5rem; color: #94a3b8; margin-top: 10px; }}
        
        .social-links {{ margin-top: 20px; }}
        .social-links a {{
            margin: 0 10px;
            padding: 8px 16px;
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 5px;
            color: white;
        }}
        .social-links a:hover {{ background: var(--primary-color); border-color: var(--primary-color); }}

        .container {{
            max-width: 1100px;
            margin: auto;
            padding: 2rem;
        }}

        section {{
            margin-bottom: 4rem;
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        }}
        
        h2 {{
            color: var(--primary-color);
            border-bottom: 2px solid #e2e8f0;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}

        .project-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}

        .project-card {{
            border: 1px solid #e2e8f0;
            padding: 20px;
            border-radius: 8px;
            transition: transform 0.2s;
            background: #fff;
        }}
        .project-card:hover {{ transform: translateY(-5px); box-shadow: 0 10px 15px rgba(0,0,0,0.1); }}
        .project-card h3 {{ margin-top: 0; }}
        .tech-stack {{
            display: inline-block;
            background: #e0f2fe;
            color: #0284c7;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9rem;
            margin-top: 10px;
            font-weight: bold;
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            background: #1e293b;
            color: #64748b;
            margin-top: 2rem;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{name}</h1>
        <p>{role}</p>
        <div class="social-links">
            <a href="mailto:{email}">Email Me</a>
            <a href="https://www.linkedin.com" target="_blank">LinkedIn</a>
            <a href="https://www.github.com" target="_blank">GitHub</a>
        </div>
    </header>

    <div class="container">
        <section id="about">
            <h2>About Me</h2>
            <p>{bio}</p>
        </section>

        <section id="projects">
            <h2>My Projects</h2>
            <div class="project-grid">
                {projects}
            </div>
        </section>
    </div>

    <footer>
        <p>&copy; 2026 {name}. Built with ResumAI.</p>
    </footer>
</body>
</html>
    """
    return html_template

# 5. Interface Layout
with st.sidebar:
    st.title("ResumAI Controls")
    st.success("⚡ Speed Mode: Enabled")
    
    if st.button("🗑️ Reset All Data", type="primary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    st.subheader("🕒 History")
    if 'history' in st.session_state and st.session_state['history']:
        for item in st.session_state['history']:
            st.text(f"• {item}")
    else:
        st.write("No actions yet.")

st.markdown("<h1 style='text-align: center;'>ResumAI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your AI-Powered Career Assistant</p>", unsafe_allow_html=True)
st.markdown("---")

selected_option = st.selectbox(
    "Select a Tool:",
    ["Select an option...", "Resume Builder", "Portfolio Builder", "Cover Letter Generator", "ATS Scanner"]
)
st.markdown("---")

# 6. Application Logic
if selected_option == "Resume Builder":
    st.header("📝 Resume Builder")
    st.subheader("1. Select Format")
    resume_style = st.selectbox(
        "Choose a Resume Template",
        [
            "Minimalist Clean (ATS Friendly)", 
            "Structured Professional (Project Focused)",
            "Chronological (Experience Focused)"
        ]
    )
    st.info(f"Selected Style: {resume_style}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name", value=st.session_state['resume_data']['name'])
        email = st.text_input("Email", value=st.session_state['resume_data']['email'])
        phone = st.text_input("Phone Number", value=st.session_state['resume_data']['phone'])
    with col2:
        role = st.text_input("Target Role", value=st.session_state['resume_data']['role'])
        linkedin = st.text_input("LinkedIn URL", value=st.session_state['resume_data']['linkedin'])
        github = st.text_input("GitHub URL", value=st.session_state['resume_data']['github'])

    skills = st.text_area("Skills (comma separated)", value=st.session_state['resume_data']['skills'])
    
    st.subheader("Experience Details")
    experience = st.text_area("Work Experience (Company, Role, Duration, Details)", value=st.session_state['resume_data']['experience'], height=150)
    
    st.subheader("Project Experience")
    projects = st.text_area("Projects (Name, Tech Stack, Description)", value=st.session_state['resume_data']['projects'], height=150)
    
    st.subheader("Education")
    education = st.text_area("Education Details (University, Degree, Year)", value=st.session_state['resume_data']['education'])
    
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 3])
    
    with btn_col1:
        if st.button("💾 Save Details"):
            st.session_state['resume_data'] = {
                'name': name, 'email': email, 'phone': phone, 'role': role,
                'linkedin': linkedin, 'github': github,
                'skills': skills, 'experience': experience, 'projects': projects, 'education': education
            }
            st.success("Details saved!")
            
    with btn_col2:
        generate_btn = st.button("✨ Generate Resume")

    if generate_btn:
        if name and role:
            base_prompt = f"""
            Write a professional resume for {name}, applying for a {role} position.
            Contact: {email}, {phone}, LinkedIn: {linkedin}, GitHub: {github}
            TECHNICAL SKILLS: {skills}
            WORK EXPERIENCE: {experience}
            PROJECT EXPERIENCE: {projects}
            EDUCATION: {education}
            """
            
            if "Minimalist Clean" in resume_style:
                style_instruction = """
                STRICT FORMAT: Minimalist (Linear).
                1. Header: Name (Bold), Role, Contact Info (in one line if possible).
                2. Summary: 2-3 lines max.
                3. Education: University, Degree, Year (Clean list).
                4. Projects: Project Name | Tech Stack | Date. Bullet points for results.
                5. Skills: Categorized list (e.g., Languages, Tools).
                NO complex tables. Clean, scannable text.
                """
            elif "Structured Professional" in resume_style:
                style_instruction = """
                STRICT FORMAT: Structured (Emphasis on Projects).
                1. Header.
                2. Professional Summary.
                3. Projects: DETAILED. Use "Project Name | Tech Stack" as header. 3 bullet points per project emphasizing numbers/results.
                4. Technical Skills: Grouped clearly.
                5. Education.
                6. Languages/Interests (Short).
                """
            else:
                style_instruction = "Standard Chronological format. Experience first, then Projects."

            final_prompt = base_prompt + "\n" + style_instruction + "\nEnsure clear headings using ## for sections."

            # --- STREAMING IMPLEMENTATION ---
            st.subheader("Preview")
            response_container = st.empty()
            full_text = ""
            
            try:
                stream_response = get_gemini_response(final_prompt, stream=True)
                
                # Check if it failed and returned a string error
                if isinstance(stream_response, str):
                    st.error(stream_response)
                else:
                    for chunk in stream_response:
                        full_text += chunk.text
                        response_container.markdown(full_text + "▌")
                    
                    # Final clean display
                    response_container.markdown(full_text)
                    
                    # Generate PDF after full text is ready
                    pdf_file = create_pdf(full_text)
                    clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
                    
                    st.download_button(
                        label="Download Resume as PDF",
                        data=pdf_file,
                        file_name=f"{clean_name}_Resume.pdf",
                        mime="application/pdf"
                    )
                    add_to_history("Generated Resume", f"{name} ({resume_style})")
                    
            except Exception as e:
                st.error(f"Error during generation: {e}")
        else:
            st.warning("Please fill in at least Name and Target Role.")

elif selected_option == "Portfolio Builder":
    st.header("💼 Portfolio Builder")
    st.markdown("Generates a **standard, editable HTML file**.")
    
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Your Name", value=st.session_state['resume_data']['name'])
        p_role = st.text_input("Current Role", value=st.session_state['resume_data']['role'])
    with col2:
        p_linkedin = st.text_input("LinkedIn Link", value=st.session_state['resume_data']['linkedin'])
        p_github = st.text_input("GitHub Link", value=st.session_state['resume_data']['github'])

    p_bio = st.text_area("About Me (Bio)", value="I am a passionate developer skilled in Python and Data Science...")
    p_raw_projects = st.text_area("List your Projects (We will format them)", value=st.session_state['resume_data']['projects'], height=150)
    
    if st.button("Generate Website Code"):
        with st.spinner("Architecting your website..."):
            
            project_prompt = f"""
            Convert these project descriptions into clean HTML <div> cards.
            Input Projects: {p_raw_projects}
            
            Output format for EACH project:
            <div class="project-card">
                <h3>Project Name</h3>
                <p>Short description...</p>
                <div class="tech-stack">Tech: Python, SQL (Example)</div>
            </div>
            
            Strictly return ONLY the HTML code. Do NOT include markdown backticks.
            """
            
            # --- STREAMING IMPLEMENTATION FOR CODE ---
            st.subheader("Generating Code...")
            code_container = st.empty()
            full_code = ""
            
            try:
                stream_response = get_gemini_response(project_prompt, stream=True)
                
                if isinstance(stream_response, str):
                    st.error(stream_response)
                else:
                    for chunk in stream_response:
                        full_code += chunk.text
                        code_container.code(full_code + "▌", language='html')
                    
                    code_container.code(full_code, language='html')
                    
                    # Clean and Assemble
                    formatted_projects = clean_ai_response(full_code)
                    full_html = generate_portfolio_html(
                        p_name, p_role, p_bio, formatted_projects, 
                        st.session_state['resume_data']['email'], 
                        p_linkedin, p_github
                    )
                    
                    st.success("Website Generated Successfully!")
                    st.components.v1.html(full_html, height=500, scrolling=True)
                    
                    st.download_button(
                        label="📥 Download index.html",
                        data=full_html,
                        file_name="index.html",
                        mime="text/html"
                    )
                    add_to_history("Generated Portfolio", p_name)
            except Exception as e:
                st.error(f"Error: {e}")

elif selected_option == "Cover Letter Generator":
    st.header("✉️ Cover Letter Generator")
    c_name = st.text_input("Your Name", value=st.session_state['resume_data']['name'])
    c_company = st.text_input("Company Name")
    c_role = st.text_input("Job Role")
    c_jd = st.text_area("Job Description Snippet")
    
    if st.button("Generate Cover Letter"):
        prompt = f"""
        Write a professional cover letter for {c_name} to {c_company} for the role of {c_role}.
        JOB DESCRIPTION: {c_jd}
        
        STRICT RULES:
        1. Tone: Professional, enthusiastic, and confident.
        2. Do NOT use brackets or placeholders like [Your Name], [Date], or [Manager Name].
        3. Use strictly the information provided. If a detail (like address) is missing, do not include that line.
        4. Sign off with the user's actual name: {c_name}.
        """
        
        # --- STREAMING IMPLEMENTATION ---
        st.subheader("Drafting Letter...")
        letter_container = st.empty()
        full_letter = ""
        
        try:
            stream_response = get_gemini_response(prompt, stream=True)
            
            if isinstance(stream_response, str):
                st.error(stream_response)
            else:
                for chunk in stream_response:
                    full_letter += chunk.text
                    letter_container.markdown(full_letter + "▌")
                
                letter_container.markdown(full_letter)
                
                pdf_file = create_pdf(full_letter)
                st.download_button(
                    label="Download Cover Letter PDF",
                    data=pdf_file,
                    file_name="Cover_Letter.pdf",
                    mime="application/pdf"
                )
                add_to_history("Generated Cover Letter", c_company)
        except Exception as e:
            st.error(f"Error: {e}")

elif selected_option == "ATS Scanner":
    st.header("🔍 ATS Scanner")
    st.info("Hybrid Analysis: TF-IDF (Math) + Gemini (AI)")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")
    ats_jd = st.text_area("Paste Job Description")
    
    if st.button("Scan Resume"):
        if uploaded_file and ats_jd:
            with st.spinner("Analyzing..."):
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                # 1. Math Analysis (Fast)
                resume_text = extract_text_from_pdf(uploaded_file)
                score = calculate_ats_score(resume_text, ats_jd)
                
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("ATS Match Score", f"{score}%")
                with col_b:
                    if score < 50:
                        st.error("Low Match - Needs Work")
                    elif score < 75:
                        st.warning("Average Match - Optimize Keywords")
                    else:
                        st.success("High Match - Ready to Apply")
                
                # 2. AI Feedback (Streaming)
                st.subheader("AI Analysis")
                feedback_prompt = f"""
                Analyze this resume against the JD.
                Resume: {resume_text}
                JD: {ats_jd}
                Provide: 1. Missing Keywords 2. Improvement Tips.
                """
                
                feedback_container = st.empty()
                full_feedback = ""
                
                try:
                    stream_response = get_gemini_response(feedback_prompt, stream=True)
                    if isinstance(stream_response, str):
                        st.error(stream_response)
                    else:
                        for chunk in stream_response:
                            full_feedback += chunk.text
                            feedback_container.write(full_feedback + "▌")
                        feedback_container.write(full_feedback)
                        
                        add_to_history("ATS Scan Performed", f"Score: {score}%")
                except Exception as e:
                    st.error(f"AI Error: {e}")
        else:
            st.error("Please upload a file and provide a JD.")