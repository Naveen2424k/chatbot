import os
import re
import json
import logging
from typing import Dict, Any, Optional
import pypdf
import docx2txt
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI

logger = logging.getLogger("ResumeParser")

class CandidateMetadata(BaseModel):
    name: str = Field(default="Unknown", description="The full name of the candidate. If not found, use 'Unknown'.")
    email: str = Field(default="", description="The email address of the candidate. Extract email, e.g. candidate@example.com. If not found, use ''.")
    skills: list = Field(default_factory=list, description="A list of technical skills, programming languages, tools, frameworks, and methodologies.")
    experience_years: float = Field(default=0.0, description="Total number of years of professional experience as a float. Sum the durations of job roles. Convert months to fractions of a year (e.g. 6 months = 0.5). If not clear, make a reasonable estimate.")
    experience_details: str = Field(default="", description="Summary of candidate's work history, including companies worked for, roles, and major projects.")
    education: str = Field(default="", description="Summary of education, including degrees, majors, and universities.")
    resume_score: int = Field(default=70, description="Resume quality score from 0 to 100 based on structure, formatting, clarity, and impact of content.")
    summary: str = Field(default="", description="A brief 2-3 sentence executive summary of the candidate's profile and core strengths.")

def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file using pypdf."""
    text = ""
    try:
        reader = pypdf.PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path}: {e}")
    return text

def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a DOCX file using docx2txt."""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {e}")
        return ""

def extract_text(file_path: str) -> str:
    """Helper function to extract text based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    else:
        logger.warning(f"Unsupported file format: {ext}")
        return ""

def parse_resume_metadata(text: str, api_key: str) -> Dict[str, Any]:
    """
    Uses Gemini to parse candidate metadata from resume text.
    Includes a fallback parser if structured output fails.
    """
    if not text.strip():
        logger.warning("Empty text passed to metadata parser.")
        return CandidateMetadata().dict()

    try:
        # Initialize Gemini via LangChain
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
        
        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(CandidateMetadata)
            prompt = f"""
            You are an expert resume parser. Extract candidate metadata from the following resume text.
            
            CRITICAL INSTRUCTIONS:
            - **Candidate Name**: Look at the very top header/first few lines of the resume text. The candidate's name is usually the first text block, in the largest font. Do not mistake headings, companies worked for, certifications, or titles (like 'Java Developer') for the candidate's name. Only return 'Unknown' if absolutely no name is present.
            - **Email**: Extract the candidate's contact email.
            
            Resume text:
            {text}
            """
            result = structured_llm.invoke(prompt)
            if result:
                return result.dict()
        except Exception as se:
            logger.warning(f"Structured output parsing failed: {se}. Falling back to prompt JSON extraction.")

        # Fallback: Prompts Gemini to return pure JSON
        prompt = f"""
        Analyze the following resume text and extract candidate details as a clean JSON object. 
        
        CRITICAL INSTRUCTIONS:
        - **Candidate Name**: Look at the very top/first few lines of the resume text. The candidate's name is usually the first text block, in the largest font. Do not mistake headings, companies worked for, certifications, or titles (like 'Java Developer') for the candidate's name. Only return 'Unknown' if absolutely no name is present.
        
        Ensure you only return valid JSON matching this schema:
        {{
            "name": "Candidate Full Name (use 'Unknown' if not found)",
            "email": "Candidate email (use '' if not found)",
            "skills": ["List of skills, technologies, tools, etc."],
            "experience_years": 3.5, // Total years of professional experience as a float (estimate if not explicit)
            "experience_details": "Summary of candidate's work history, roles, companies",
            "education": "Summary of candidate's degrees, majors, universities",
            "resume_score": 75, // Integer 0-100 evaluating the resume's clarity and content
            "summary": "Brief 2-3 sentence executive summary of the candidate's profile"
        }}
        
        Return ONLY the raw JSON block. Do not include markdown code block syntax (like ```json) or any other text.
        
        Resume text:
        {text}
        """
        
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean response if markdown blocks are returned
        if response_text.startswith("```"):
            # strip markdown lines
            lines = response_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
            
        data = json.loads(response_text)
        
        # Validate through CandidateMetadata Pydantic model to fill defaults
        validated_data = CandidateMetadata(**data)
        return validated_data.dict()

    except Exception as e:
        logger.error(f"Failed to parse resume metadata with Gemini: {e}")
        # Return fallback default values
        return CandidateMetadata().dict()
