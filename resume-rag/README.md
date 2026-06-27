# 💼 AI Resume Analyzer and RAG Chatbot

A production-ready recruiter dashboard, resume analyzer, and conversational RAG chatbot. Built using **Streamlit**, **LangChain**, **ChromaDB**, and **Gemini**.

## 🚀 Key Features

1. **Multiple Resume Uploads**: Ingest and process multiple PDF/DOCX resumes simultaneously.
2. **AI-Powered Structured Parsing**: Extracts candidate information (Name, Email, Skills, Experience Years, Education, Resume Quality Score, Executive Summary) into a relational database.
3. **Conversational RAG Chatbot**: Chat with your resumes to find candidates using queries like *"Who has Java & Spring Boot experience?"* or *"List candidates with MongoDB experience"*. Maintain context through conversation history memory.
4. **Candidate Leaderboard**: Display candidate metrics, filter profiles, and download the entire parsed directory as a CSV spreadsheet.
5. **Job Description Matching (ATS)**: Upload a Job Description (JD) PDF, compute match percentages, generate ATS scores, isolate strengths/gaps, and rank candidates automatically.
6. **Robust Hybrid Data Storage**:
   - **SQLite (`database/metadata.db`)** manages relational search indexes, candidate profiles, and match scores.
   - **ChromaDB (`chroma_db/`)** manages document chunk embeddings for high-speed semantic retrieval.

---

## 🛠️ Project Structure

```text
resume-rag/
├── app.py                  # Streamlit application UI & Dashboard
├── rag/
│   ├── embeddings.py       # Embeddings initializer (text-embedding-004)
│   ├── retriever.py        # RAG pipeline combining SQLite metadata & Chroma chunks
│   ├── parser.py           # Text extractors (PDF/DOCX) & structured Gemini parser
│   ├── ranker.py           # Job Description comparative scoring
│   └── vectorstore.py      # ChromaDB document chunking, insert, & delete operations
├── database/
│   └── db_manager.py       # SQLite relational database manager
├── uploads/                # Directory for storing raw uploaded resumes & JDs
├── chroma_db/              # Directory containing ChromaDB vector storage files
├── requirements.txt        # Third-party python dependencies
└── README.md               # Setup and usage guide
```

---

## ⚙️ Installation & Setup

### Pre-requisites
- **Python 3.10+** installed.
- A **Google Gemini API Key** (from [Google AI Studio](https://aistudio.google.com/)).

### 1. Clone & Navigate
Place the files into your project workspace folder `resume-rag/`.

### 2. Create a Virtual Environment
```bash
python -m venv venv
# On Windows (PowerShell)
.\venv\Scripts\Activate.ps1
# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a file named `.env` in the root folder `resume-rag/`:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key_here
```
*Note: If the `.env` file is missing, you can enter the Google API Key directly into the sidebar text input in the app UI.*

### 5. Launch the Application
```bash
streamlit run app.py
```
The application will open automatically in your browser (usually at `http://localhost:8501`).

---

## 💡 Key Usage Walkthrough

1. **Upload Resumes**: Drag and drop resumes in PDF or DOCX format into the sidebar. Click **Process & Index Resumes**.
2. **Check Leaderboard**: Navigate to the **Candidate Leaderboard** tab to view the metrics table. Click **Download Leaderboard as CSV** to export.
3. **Chat**: Select the **Recruiter Chat** tab. Type queries like *"Find Python developers with Django experience"* or *"List candidates with over 3 years experience"*.
4. **ATS JD Matching**: Upload a Job Description PDF in the sidebar. Click **Match Candidates against JD**. Navigate to **ATS Match Analytics** to view comparative bar charts and matching strengths/gaps for each applicant.

---

## 🐳 Deployment Guide

### Deploy to Streamlit Community Cloud
1. Push this project folder to a GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select your repository, branch, and `app.py` as the entry file.
4. Under **Settings -> Secrets**, define the API Key:
   ```toml
   GOOGLE_API_KEY = "your_gemini_api_key"
   ```
5. Click **Deploy**.
