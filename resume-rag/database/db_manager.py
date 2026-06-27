import os
import sqlite3
import logging
from typing import List, Dict, Any, Optional

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DBManager")

class DBManager:
    """Manages the SQLite database for candidate metadata and ATS scoring."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Place database/metadata.db in the directory of this file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, "metadata.db")
        else:
            self.db_path = db_path
            
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enables access by column name
        return conn

    def init_db(self):
        """Initializes the database schema if it doesn't exist."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS candidates (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        email TEXT,
                        skills TEXT,
                        experience_years REAL,
                        experience_details TEXT,
                        education TEXT,
                        resume_score INTEGER,
                        summary TEXT,
                        resume_path TEXT,
                        ats_score REAL,
                        match_percentage REAL,
                        jd_analysis TEXT
                    )
                """)
                conn.commit()
                logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise e

    def add_or_update_candidate(self, candidate_data: Dict[str, Any]) -> bool:
        """
        Inserts a new candidate or updates an existing one if the ID already exists.
        
        Args:
            candidate_data: Dictionary containing candidate fields.
        """
        sql = """
            INSERT INTO candidates (
                id, name, email, skills, experience_years, 
                experience_details, education, resume_score, 
                summary, resume_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name=excluded.name,
                email=excluded.email,
                skills=excluded.skills,
                experience_years=excluded.experience_years,
                experience_details=excluded.experience_details,
                education=excluded.education,
                resume_score=excluded.resume_score,
                summary=excluded.summary,
                resume_path=excluded.resume_path
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (
                    candidate_data['id'],
                    candidate_data.get('name', 'Unknown'),
                    candidate_data.get('email', ''),
                    candidate_data.get('skills', ''),
                    candidate_data.get('experience_years', 0.0),
                    candidate_data.get('experience_details', ''),
                    candidate_data.get('education', ''),
                    candidate_data.get('resume_score', 0),
                    candidate_data.get('summary', ''),
                    candidate_data.get('resume_path', '')
                ))
                conn.commit()
                logger.info(f"Candidate {candidate_data.get('name')} saved successfully.")
                return True
        except Exception as e:
            logger.error(f"Error saving candidate {candidate_data.get('id')}: {e}")
            return False

    def update_jd_match(self, candidate_id: str, ats_score: float, match_percentage: float, jd_analysis: str) -> bool:
        """Updates the JD match scores and details for a candidate."""
        sql = """
            UPDATE candidates 
            SET ats_score = ?, match_percentage = ?, jd_analysis = ?
            WHERE id = ?
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, (ats_score, match_percentage, jd_analysis, candidate_id))
                conn.commit()
                logger.info(f"JD Match updated for candidate {candidate_id}.")
                return True
        except Exception as e:
            logger.error(f"Error updating JD match for candidate {candidate_id}: {e}")
            return False

    def get_all_candidates(self) -> List[Dict[str, Any]]:
        """Retrieves all candidates from the database."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM candidates")
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error retrieving candidates: {e}")
            return []

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single candidate by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM candidates WHERE id = ?", (candidate_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            logger.error(f"Error retrieving candidate {candidate_id}: {e}")
            return None

    def delete_candidate(self, candidate_id: str) -> bool:
        """Deletes a candidate by ID."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM candidates WHERE id = ?", (candidate_id,))
                conn.commit()
                logger.info(f"Candidate {candidate_id} deleted from database.")
                return True
        except Exception as e:
            logger.error(f"Error deleting candidate {candidate_id}: {e}")
            return False

    def clear_database(self) -> bool:
        """Deletes all candidate records."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM candidates")
                conn.commit()
                logger.info("All records cleared from database.")
                return True
        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def clear_jd_results(self) -> bool:
        """Resets all JD match scores to NULL."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE candidates 
                    SET ats_score = NULL, match_percentage = NULL, jd_analysis = NULL
                """)
                conn.commit()
                logger.info("JD match scores cleared for all candidates.")
                return True
        except Exception as e:
            logger.error(f"Error clearing JD match scores: {e}")
            return False
