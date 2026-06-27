import os
import logging
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Import our RAG modules
from database.db_manager import DBManager
from rag.parser import extract_text, parse_resume_metadata
from rag.vectorstore import add_resume_to_vector_store, clear_vector_store
from rag.ranker import match_resume_to_jd
from rag.retriever import get_rag_response

# Load environment variables
load_dotenv()

# Initialize Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("App")

def clean_filename_to_name(filename: str) -> str:
    """Fallback utility to extract candidate name from file name if parser fails."""
    import re
    # remove extension
    base = os.path.splitext(filename)[0]
    # replace punctuation and common words with space
    base = re.sub(r'(?i)(resume|cv|updated|final|latest|job|application|_|\-)', ' ', base)
    # strip numbers
    base = re.sub(r'\d+', ' ', base)
    # clean extra spaces
    name = " ".join(base.split()).strip()
    return name.title() if name else "Unknown Candidate"


# App Setup
st.set_page_config(
    page_title="AI Resume Analyzer & RAG Chatbot",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #4A154B 0%, #6C63FF 50%, #0F0C20 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(108, 99, 255, 0.25);
        text-align: center;
    }
    .main-header h1 {
        font-weight: 700;
        margin: 0;
        font-size: 2.6rem;
        background: linear-gradient(to right, #ffffff, #e0e0ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        font-weight: 300;
        margin-top: 0.5rem;
        font-size: 1.1rem;
        color: #e2d9f3;
    }
    
    /* Card Styles */
    .metric-card {
        background: #1E1E38;
        border: 1px solid #3F3D56;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: #6C63FF;
    }
    .metric-val {
        font-size: 2.2rem;
        font-weight: 700;
        color: #6C63FF;
        margin-bottom: 0.2rem;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #B3B3D4;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Sidebar styling tweaks */
    .sidebar .sidebar-content {
        background-color: #0E1117;
    }
    
    /* Subheaders styling */
    .section-title {
        font-weight: 600;
        color: #6C63FF;
        border-bottom: 2px solid #6C63FF;
        padding-bottom: 0.3rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SESSION STATE -----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Paths configuration
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOADS_DIR = os.path.join(CURRENT_DIR, "uploads")
DB_PATH = os.path.join(CURRENT_DIR, "database", "metadata.db")
CHROMA_DIR = os.path.join(CURRENT_DIR, "chroma_db")

os.makedirs(UPLOADS_DIR, exist_ok=True)

# Initialize DB Manager
db_manager = DBManager(DB_PATH)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/resume.png", width=80)
    st.title("Settings & Uploads")
    
    # 1. API Key Input
    api_key_env = os.environ.get("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google API Key", 
        type="password", 
        value=api_key_env,
        help="Provide your Gemini API key here if it's not set in the environment variables."
    )
    
    api_key = api_key_input if api_key_input else api_key_env
    
    if not api_key:
        st.warning("⚠️ Please provide a Google API Key to enable AI features.")

    st.markdown("---")
    
    # 2. Resumes Upload
    st.subheader("1. Ingest Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload Resumes (PDF, DOCX)", 
        type=["pdf", "docx"], 
        accept_multiple_files=True,
        key="resume_uploader"
    )
    
    if st.button("Process & Index Resumes", disabled=not api_key or not uploaded_resumes, use_container_width=True):
        success_count = 0
        progress_bar = st.progress(0.0)
        
        for idx, file in enumerate(uploaded_resumes):
            file_path = os.path.join(UPLOADS_DIR, file.name)
            
            # Save file locally
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            # Step 1: Extract Text
            text = extract_text(file_path)
            
            if text.strip():
                # Step 2: Use LLM to extract metadata
                with st.spinner(f"Analyzing {file.name}..."):
                    metadata = parse_resume_metadata(text, api_key)
                
                # Enrich metadata
                metadata['id'] = file.name  # Use filename as unique ID
                metadata['resume_path'] = file_path
                
                # Fallback if name is Unknown or empty
                if not metadata.get('name') or metadata.get('name').strip() == 'Unknown':
                    metadata['name'] = clean_filename_to_name(file.name)
                
                # Check for list skills conversion to string
                if isinstance(metadata.get('skills'), list):
                    metadata['skills'] = ", ".join(metadata['skills'])
                
                # Step 3: Save to SQLite
                db_success = db_manager.add_or_update_candidate(metadata)
                
                # Step 4: Add chunks to ChromaDB
                vect_success = add_resume_to_vector_store(file_path, file.name, api_key, CHROMA_DIR)
                
                if db_success and vect_success:
                    success_count += 1
            
            progress_bar.progress((idx + 1) / len(uploaded_resumes))
            
        if success_count > 0:
            st.success(f"Successfully processed {success_count} resumes!")
            # Trigger page rerun to refresh candidate stats/views
            st.rerun()
        else:
            st.error("Failed to process resumes. Verify API key and file formats.")
            
    st.markdown("---")
    
    # 3. Job Description Input
    st.subheader("2. Job Description (JD)")
    jd_input_method = st.radio(
        "JD Input Method",
        ["Upload JD PDF", "Paste JD Text"],
        key="jd_input_method"
    )
    
    jd_text = ""
    uploaded_jd = None
    
    if jd_input_method == "Upload JD PDF":
        uploaded_jd = st.file_uploader("Upload JD (PDF)", type=["pdf"], key="jd_uploader")
        has_jd_source = uploaded_jd is not None
    else:
        jd_text_input = st.text_area("Paste Job Description Text:", height=150, key="jd_text_area")
        jd_text = jd_text_input.strip()
        has_jd_source = len(jd_text) > 0
    
    if st.button("Match Candidates against JD", disabled=not api_key or not has_jd_source, use_container_width=True):
        if jd_input_method == "Upload JD PDF":
            # Save JD
            jd_path = os.path.join(UPLOADS_DIR, "active_job_description.pdf")
            with open(jd_path, "wb") as f:
                f.write(uploaded_jd.getbuffer())
                
            # Extract text from JD
            jd_text = extract_text(jd_path)
        else:
            # Save plain text JD
            jd_path = os.path.join(UPLOADS_DIR, "active_job_description.txt")
            with open(jd_path, "w", encoding="utf-8") as f:
                f.write(jd_text)
            
        if jd_text.strip():
            candidates = db_manager.get_all_candidates()
            
            if not candidates:
                st.warning("No candidates found in the database. Please upload resumes first.")
            else:
                progress_bar = st.progress(0.0)
                match_count = 0
                
                for idx, cand in enumerate(candidates):
                    resume_text = extract_text(cand['resume_path'])
                    
                    with st.spinner(f"Matching {cand['name']} against JD..."):
                        match_details = match_resume_to_jd(resume_text, jd_text, api_key)
                        
                    # Save back to SQLite
                    db_success = db_manager.update_jd_match(
                        cand['id'], 
                        match_details.get('ats_score', 0.0),
                        match_details.get('match_percentage', 0.0),
                        json_str := str(match_details)  # Store match detail dictionary representation
                    )
                    if db_success:
                        match_count += 1
                    
                    progress_bar.progress((idx + 1) / len(candidates))
                    
                if match_count > 0:
                    st.success(f"Matched {match_count} candidates successfully!")
                    st.rerun()
                else:
                    st.error("Could not run JD comparison matches.")
        else:
            st.error("Could not extract text from the provided Job Description.")
            
    st.markdown("---")
    
    # 4. Management & Reset Utilities
    st.subheader("System Tools")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear DB", type="secondary", use_container_width=True):
            db_manager.clear_database()
            if api_key:
                clear_vector_store(api_key, CHROMA_DIR)
            # Remove uploaded files
            for file in os.listdir(UPLOADS_DIR):
                os.remove(os.path.join(UPLOADS_DIR, file))
            st.session_state.messages = []
            st.success("All data cleared.")
            st.rerun()
    with col2:
        if st.button("Reset JD Match", type="secondary", use_container_width=True):
            db_manager.clear_jd_results()
            st.success("JD results reset.")
            st.rerun()

# ----------------- MAIN INTERFACE -----------------

# Gradient Header
st.markdown("""
<div class="main-header">
    <h1>💼 AI Resume Analyzer & RAG Chatbot</h1>
    <p>Upload resumes, search semantic patterns, compare matches with JDs, and chat interactively about candidates.</p>
</div>
""", unsafe_allow_html=True)

# Fetch all candidates from DB
all_candidates = db_manager.get_all_candidates()

# Display quick metrics
if all_candidates:
    df_metrics = pd.DataFrame(all_candidates)
    total_cands = len(df_metrics)
    max_score = int(df_metrics['resume_score'].max())
    
    # Calculate JD Stats if matched
    has_jd_matches = 'match_percentage' in df_metrics.columns and df_metrics['match_percentage'].notna().any()
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{total_cands}</div>
            <div class="metric-title">Total Candidates</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{max_score}/100</div>
            <div class="metric-title">Max Resume Score</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        if has_jd_matches:
            avg_match = int(df_metrics['match_percentage'].dropna().mean())
            val = f"{avg_match}%"
        else:
            val = "N/A"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-title">Average JD Match</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        if has_jd_matches:
            top_rec = df_metrics.sort_values(by="match_percentage", ascending=False).iloc[0]['name']
            val = top_rec if len(top_rec) < 18 else f"{top_rec[:15]}..."
        else:
            val = "None"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="font-size: 1.5rem; line-height: 2.2rem; padding: 0.15rem 0;">{val}</div>
            <div class="metric-title">Top JD Fit</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab_chat, tab_leaderboard, tab_ats = st.tabs([
    "💬 Recruiter Chat (RAG)", 
    "🏆 Candidate Leaderboard", 
    "📊 ATS Match Analytics"
])

# ----------------- TAB 1: RECRUITER CHAT (RAG) -----------------
with tab_chat:
    st.markdown('<h3 class="section-title">Ask Questions about Resumes</h3>', unsafe_allow_html=True)
    
    if not all_candidates:
        st.info("💡 To start, upload and process some resumes in the sidebar!")
    else:
        # Chat container
        chat_container = st.container(height=500)
        with chat_container:
            if not st.session_state.messages:
                st.markdown("""
                👋 **Welcome to the recruiter assistant!** I can help you search and analyze candidate resumes.
                Try asking questions like:
                * *"Find candidates with Java and Spring Boot experience."*
                * *"Who has more than 3 years of experience in data analytics?"*
                * *"List candidate profiles including name, email, skills, and summary."*
                * *"Compare resumes and summarize MongoDB skills."*
                """)
                
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Search or ask questions..."):
            if not api_key:
                st.error("Please configure your Gemini API Key in the sidebar first.")
            else:
                # Add user input
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                    # Get response from retriever
                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing and retrieving context..."):
                            response = get_rag_response(
                                query=prompt, 
                                chat_history=st.session_state.messages[:-1], 
                                api_key=api_key, 
                                db_path=DB_PATH, 
                                chroma_dir=CHROMA_DIR
                            )
                            st.markdown(response)
                
                # Append assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                
        # Clear chat option
        if st.session_state.messages:
            if st.button("Clear Conversation"):
                st.session_state.messages = []
                st.rerun()

# ----------------- TAB 2: CANDIDATE LEADERBOARD -----------------
with tab_leaderboard:
    st.markdown('<h3 class="section-title">Candidate Profiles & Metric Overview</h3>', unsafe_allow_html=True)
    
    if not all_candidates:
        st.warning("No candidate records parsed yet. Ingest resumes from the sidebar.")
    else:
        # Prepare Leaderboard Dataframe
        df = pd.DataFrame(all_candidates)
        
        # Select columns to show
        show_cols = ['name', 'email', 'skills', 'experience_years', 'resume_score', 'summary']
        
        # Add JD columns if available
        is_jd_available = 'match_percentage' in df.columns and df['match_percentage'].notna().any()
        if is_jd_available:
            show_cols = ['name', 'match_percentage', 'ats_score'] + [c for c in show_cols if c not in ['name']]
            # Sort by match percentage by default
            df_sorted = df.sort_values(by="match_percentage", ascending=False)
        else:
            # Sort by resume score by default
            df_sorted = df.sort_values(by="resume_score", ascending=False)
            
        # Clean up column names for displaying
        display_df = df_sorted[show_cols].copy()
        display_df.columns = [col.replace("_", " ").title() for col in display_df.columns]
        
        # Display table
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "Match Percentage": st.column_config.NumberColumn(format="%.1f%%"),
                "Ats Score": st.column_config.NumberColumn(format="%.1f/100"),
                "Resume Score": st.column_config.NumberColumn(format="%d/100"),
                "Experience Years": st.column_config.NumberColumn(format="%.1f years")
            }
        )
        
        # Download Button
        csv_data = df_sorted.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Leaderboard as CSV",
            data=csv_data,
            file_name="candidate_leaderboard.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Details view for individual candidate
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Detailed Candidate Profile Viewer")
        selected_cand_name = st.selectbox("Select a candidate to view full details:", df_sorted['name'].unique())
        
        if selected_cand_name:
            cand_detail = df_sorted[df_sorted['name'] == selected_cand_name].iloc[0]
            
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(f"**Name:** {cand_detail['name']}")
                st.markdown(f"**Email:** {cand_detail['email']}")
                st.markdown(f"**Total Experience (Years):** {cand_detail['experience_years']} years")
                st.markdown(f"**Education:** {cand_detail['education']}")
            with col_d2:
                st.markdown(f"**Resume Quality Score:** {cand_detail['resume_score']}/100")
                if is_jd_available and pd.notna(cand_detail['match_percentage']):
                    st.markdown(f"**JD Match Percentage:** {cand_detail['match_percentage']}%")
                    st.markdown(f"**JD ATS Fit Score:** {cand_detail['ats_score']}/100")
                st.markdown(f"**Document Filename:** {cand_detail['id']}")
            
            st.markdown(f"**Executive Profile Summary:**")
            st.info(cand_detail['summary'])
            
            st.markdown(f"**Parsed Key Skills:**")
            st.write(cand_detail['skills'])
            
            st.markdown(f"**Experience Details:**")
            st.write(cand_detail['experience_details'])

# ----------------- TAB 3: ATS MATCH ANALYTICS -----------------
with tab_ats:
    st.markdown('<h3 class="section-title">ATS & JD Compatibility Analytics</h3>', unsafe_allow_html=True)
    
    if not all_candidates:
        st.info("💡 Upload resumes and matching Job Description to view analytics here.")
    else:
        df = pd.DataFrame(all_candidates)
        is_jd_available = 'match_percentage' in df.columns and df['match_percentage'].notna().any()
        
        if not is_jd_available:
            st.warning("⚠️ No Job Description match has been run yet. Upload a JD PDF in the sidebar and match candidates.")
        else:
            # Graph visualization
            df_match = df.dropna(subset=['match_percentage']).sort_values(by="match_percentage", ascending=False)
            
            # Create side-by-side comparison charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.subheader("Match Percentage Comparison")
                st.bar_chart(data=df_match, x="name", y="match_percentage", color="#6C63FF")
            with chart_col2:
                st.subheader("ATS Fit Score Comparison")
                st.bar_chart(data=df_match, x="name", y="ats_score", color="#4A154B")
                
            # Detail section for match reports
            st.markdown("---")
            st.subheader("Candidate Match & Fit Reports")
            
            for index, row in df_match.iterrows():
                try:
                    # Parse the string details back to dict
                    # Note: in sqlite we saved str(match_details). It's formatted like a dict.
                    # We can use eval or safe parsing to recreate dict.
                    import ast
                    match_info = ast.literal_eval(row['jd_analysis'])
                except Exception as eval_err:
                    # Fallback representation if parsing fails
                    match_info = {
                        "recommendation": "N/A",
                        "explanation": row.get("jd_analysis", "No details"),
                        "strengths": [],
                        "gaps": []
                    }
                    
                rec_color = {
                    "Strongly Recommend": "🟢 Strongly Recommend",
                    "Recommend": "🔵 Recommend",
                    "Consider": "🟡 Consider",
                    "Not Suitable": "🔴 Not Suitable"
                }.get(match_info.get('recommendation'), "⚪ " + str(match_info.get('recommendation')))
                
                with st.expander(f"📋 {row['name']} — Match: {row['match_percentage']}% | Fit Score: {row['ats_score']}/100"):
                    st.markdown(f"#### Recommendation: {rec_color}")
                    st.write(match_info.get('explanation', ''))
                    
                    col_str, col_gap = st.columns(2)
                    with col_str:
                        st.markdown("**⭐ Core Strengths & Matches:**")
                        strengths = match_info.get('strengths', [])
                        if strengths:
                            for s in strengths:
                                st.markdown(f"- {s}")
                        else:
                            st.write("*No notable strengths extracted.*")
                    with col_gap:
                        st.markdown("**⚠️ Qualifications & Keyword Gaps:**")
                        gaps = match_info.get('gaps', [])
                        if gaps:
                            for g in gaps:
                                st.markdown(f"- {g}")
                        else:
                            st.write("*No key gaps identified.*")
