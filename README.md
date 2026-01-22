# ResumAI - AI-Powered Career Assistant
ResumAI is a comprehensive Streamlit application designed to streamline the job application process. Leveraging the power of Google Gemini 2.0 Flash, it acts as a personal career assistant to generate professional resumes, build portfolio websites, draft cover letters, and optimize applications for Applicant Tracking Systems (ATS).

# Features
## AI Resume Builder

Generates professional resumes based on user details (Experience, Skills, Projects).

Offers multiple templates: Minimalist Clean (ATS Friendly), Structured Professional, and Chronological.

Exports directly to PDF using reportlab.

## Portfolio Website Generator

Converts your profile and project descriptions into a clean, deployable HTML/CSS portfolio.

Uses AI to format project descriptions into responsive cards.

## Cover Letter Generator

Drafts tailored cover letters based on your profile and a specific Job Description (JD).

Ensures a professional tone and proper formatting.

## ATS Scanner

Hybrid Analysis: Combines TF-IDF (Math) scoring with Gemini (AI) qualitative analysis.

Provides a percentage match score and specific feedback on missing keywords and improvements.

## 🛠️ Tech Stack
Frontend: Streamlit

AI Engine: Google Gemini (google-genai SDK)

PDF Processing: reportlab, PyPDF2

Data Analysis: scikit-learn (for ATS scoring)

# Prerequisites
Python 3.8+

A Google Cloud Project with the Gemini API enabled.

An API Key from Google AI Studio.

# Project Structure
Code.py: The main application script containing the UI layout, AI logic, and PDF generation code.

.env: Stores sensitive environment variables (API Key).

requirements.txt: List of Python libraries required to run the app.

# Install dependencies:
pip install -r requirements.txt

# Usage
Run the Streamlit application: streamlit run Code.py
