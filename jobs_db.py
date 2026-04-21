import sqlite3
from pathlib import Path
from datetime import datetime
 
DB_PATH = Path("data/jobs.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
 
 
def init_db():
    """Create the jobs table if it does not exist."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id      TEXT PRIMARY KEY,
            status      TEXT DEFAULT 'queued',
            step        TEXT DEFAULT '',
            created_at  TEXT,
            regulation  TEXT,
            policy      TEXT,
            result      TEXT,
            error       TEXT
        )
    """)
    conn.commit()
    conn.close()
 
 
def save_job(job: dict):
    """Insert or update a job record."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT OR REPLACE INTO jobs
        (job_id, status, step, created_at, regulation, policy, result, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        job.get("job_id"),
        job.get("status", "queued"),
        job.get("step", ""),
        job.get("created_at", datetime.now().isoformat()),
        job.get("regulation", ""),
        job.get("policy", ""),
        job.get("result"),
        job.get("error"),
    ))
    conn.commit()
    conn.close()
 
 
def get_job(job_id: str) -> dict | None:
    """Fetch a single job by ID. Returns None if not found."""
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT job_id, status, step, created_at, regulation, "
        "policy, result, error FROM jobs WHERE job_id = ?",
        (job_id,)
    ).fetchone()
    conn.close()
    if not row:
        return None
    cols = ["job_id", "status", "step", "created_at",
            "regulation", "policy", "result", "error"]
    return dict(zip(cols, row))
 
 
def get_all_jobs() -> list[dict]:
    """Fetch all jobs, newest first."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT job_id, status, step, created_at, regulation, "
        "policy, result, error FROM jobs ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    cols = ["job_id", "status", "step", "created_at",
            "regulation", "policy", "result", "error"]
    return [dict(zip(cols, row)) for row in rows]
 
 
def update_job(job_id: str, **kwargs):
    """Update specific fields of a job."""
    if not kwargs:
        return
    fields = ", ".join(f"{k} = ?" for k in kwargs)
    values = list(kwargs.values()) + [job_id]
    conn   = sqlite3.connect(DB_PATH)
    conn.execute(f"UPDATE jobs SET {fields} WHERE job_id = ?", values)
    conn.commit()
    conn.close()