import json
import logging
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from rag.parser import extract_text

logger = logging.getLogger("Ranker")

class JDMatchDetail(BaseModel):
    match_percentage: float = Field(default=0.0, description="Percentage score from 0.0 to 100.0 representing how well the candidate's skills and experience match the JD.")
    ats_score: float = Field(default=0.0, description="ATS score from 0.0 to 100.0 assessing candidate suitability based on keywords alignment, formatting, and role fit.")
    strengths: List[str] = Field(default_factory=list, description="Key matching skills, experience, and qualifications.")
    gaps: List[str] = Field(default_factory=list, description="Missing keywords, experience gaps, or credentials.")
    recommendation: str = Field(default="Consider", description="One of: 'Strongly Recommend', 'Recommend', 'Consider', 'Not Suitable'.")
    explanation: str = Field(default="", description="A short summary of the candidate's alignment with the Job Description.")

def match_resume_to_jd(resume_text: str, jd_text: str, api_key: str) -> Dict[str, Any]:
    """
    Compares a candidate's resume text against the Job Description text.
    Returns match percentage, ATS score, strengths, gaps, and recommendation.
    """
    if not resume_text.strip() or not jd_text.strip():
        logger.warning("Empty resume or JD text provided for matching.")
        return JDMatchDetail().dict()

    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0
        )
        
        # Try structured output first
        try:
            structured_llm = llm.with_structured_output(JDMatchDetail)
            prompt = f"""
            You are an expert HR recruiter and ATS (Applicant Tracking System). 
            Compare the candidate's Resume against the Job Description.
            
            Job Description:
            {jd_text}
            
            Candidate Resume:
            {resume_text}
            """
            result = structured_llm.invoke(prompt)
            if result:
                return result.dict()
        except Exception as se:
            logger.warning(f"Structured output for JD ranking failed: {se}. Falling back to prompt JSON.")

        # Fallback: Prompts Gemini to return pure JSON
        prompt = f"""
        You are an expert HR recruiter and ATS (Applicant Tracking System). 
        Analyze the candidate's Resume against the Job Description and return a JSON evaluation.
        
        Job Description:
        {jd_text}
        
        Candidate Resume:
        {resume_text}
        
        Return ONLY a raw JSON object matching this schema. Do not include markdown code block syntax (like ```json) or any other text.
        Schema:
        {{
            "match_percentage": 78.5, // float from 0.0 to 100.0
            "ats_score": 82.0, // float from 0.0 to 100.0
            "strengths": ["list", "of", "strengths"],
            "gaps": ["list", "of", "missing", "skills/experience"],
            "recommendation": "Strongly Recommend", // One of: 'Strongly Recommend', 'Recommend', 'Consider', 'Not Suitable'
            "explanation": "Brief explanation of the fit"
        }}
        """
        
        response = llm.invoke(prompt)
        response_text = response.content.strip()
        
        # Clean response if markdown blocks are returned
        if response_text.startswith("```"):
            lines = response_text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            response_text = "\n".join(lines).strip()
            
        data = json.loads(response_text)
        validated_data = JDMatchDetail(**data)
        return validated_data.dict()

    except Exception as e:
        logger.error(f"Error matching resume to JD: {e}")
        return JDMatchDetail().dict()
