import logging
from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from database.db_manager import DBManager
from rag.vectorstore import get_vector_store

logger = logging.getLogger("Retriever")

def get_rag_response(
    query: str, 
    chat_history: List[Dict[str, str]], 
    api_key: str, 
    db_path: str = None, 
    chroma_dir: str = None
) -> str:
    """
    Executes the conversational RAG pipeline:
    1. Retrieval: Search ChromaDB for relevant text chunks.
    2. Candidate Lookup: Pull structured candidate profiles from SQLite database.
    3. LLM Querying: Query Gemini using structured context and conversation history.
    """
    try:
        # 1. Initialize databases
        db_manager = DBManager(db_path)
        vector_store = get_vector_store(api_key, chroma_dir) if chroma_dir else get_vector_store(api_key)
        
        # 2. Perform similarity search
        # We search Chroma for candidate chunks matching the recruiter's query
        retrieved_docs = vector_store.similarity_search(query, k=8)
        
        # 3. Compile chunks and find matching candidate IDs
        chunks_text = []
        candidate_ids = set()
        for doc in retrieved_docs:
            candidate_id = doc.metadata.get("candidate_id")
            source_file = doc.metadata.get("source", "Unknown File")
            chunks_text.append(f"[File: {source_file} | Candidate ID: {candidate_id}]\n{doc.page_content}\n")
            if candidate_id:
                candidate_ids.add(candidate_id)
                
        # 4. Fetch candidate details from SQLite
        candidate_profiles = []
        for cid in candidate_ids:
            cand = db_manager.get_candidate(cid)
            if cand:
                profile = (
                    f"- **Candidate Name**: {cand.get('name')}\n"
                    f"  - **Email**: {cand.get('email')}\n"
                    f"  - **Skills**: {cand.get('skills')}\n"
                    f"  - **Experience**: {cand.get('experience_years')} years ({cand.get('experience_details')})\n"
                    f"  - **Education**: {cand.get('education')}\n"
                    f"  - **Resume Score**: {cand.get('resume_score')}/100\n"
                    f"  - **Summary**: {cand.get('summary')}\n"
                )
                candidate_profiles.append(profile)
                
        # If the search results are empty, fetch general candidate profiles from the DB
        # so the model has some context on who is available.
        if not candidate_profiles:
            all_cands = db_manager.get_all_candidates()
            for cand in all_cands[:5]:  # Limit to 5 for context window safety
                profile = (
                    f"- **Candidate Name**: {cand.get('name')}\n"
                    f"  - **Email**: {cand.get('email')}\n"
                    f"  - **Skills**: {cand.get('skills')}\n"
                    f"  - **Experience**: {cand.get('experience_years')} years\n"
                    f"  - **Resume Score**: {cand.get('resume_score')}/100\n"
                )
                candidate_profiles.append(profile)

        # 5. Format contexts
        profiles_context = "\n".join(candidate_profiles) if candidate_profiles else "No candidate profiles found in the database."
        chunks_context = "\n---\n".join(chunks_text) if chunks_text else "No relevant resume text chunks retrieved."

        # 6. Format chat history
        history_str = ""
        for msg in chat_history:
            role = "Recruiter" if msg["role"] == "user" else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
            
        # 7. Construct system prompt
        system_prompt = f"""You are a professional HR recruiter assistant. 
Your goal is to answer questions about candidates based on their resumes and metadata.

Here are the candidate profiles from the database:
{profiles_context}

Here are matching sections extracted directly from candidate resumes:
{chunks_context}

Provide a comprehensive, professional response to the Recruiter's question.
If they ask to "find" or "list" candidates, return details of the matching candidates including Name, Email, Skills, Experience, Education, Resume Score, and Resume Summary.
Use the structured candidate profiles above as the primary source of metadata, and use the resume text chunks for specific detailed questions.
If you cannot answer the question or find matching candidates, say so politely.

Chat History:
{history_str}
Recruiter: {query}
Assistant:"""

        # 8. Call Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.2
        )
        
        response = llm.invoke(system_prompt)
        return response.content.strip()
        
    except Exception as e:
        logger.error(f"Error in RAG retrieval: {e}")
        return f"Sorry, I encountered an error while processing your request: {str(e)}"
