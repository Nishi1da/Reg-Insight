"""
REG-INSIGHT — FastAPI Backend
==============================
Run with:  uvicorn api:app --reload --port 8000
Docs at:   http://localhost:8000/docs
"""

import json
import uuid
import hashlib
import sys
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("api")

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "src"))

OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR = Path("data/raw/uploads")
OUTPUTS_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# ── Jobs DB ───────────────────────────────────────────────────────────────────
from jobs_db import init_db, save_job, get_job, get_all_jobs, update_job
init_db()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="REG-INSIGHT API",
    description="Regulatory Compliance Gap Detection — REST API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    """Check if API and key dependencies are reachable."""
    status = {
        "api":               "ok",
        "chroma":            "unknown",
        "groq":              "unknown",
        "reports_available": 0,
    }

    try:
        import chromadb
        client = chromadb.PersistentClient(path="data/processed/chroma_db")
        client.list_collections()
        status["chroma"] = "ok"
    except Exception as e:
        status["chroma"] = f"error: {str(e)[:60]}"

    if Path(".env").exists() or Path("config/groq_config.yaml").exists():
        status["groq"] = "config_found"
    else:
        status["groq"] = "config_missing"

    status["reports_available"] = len(list(OUTPUTS_DIR.glob("*.json")))

    return status


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/reports")
def list_reports():
    """List all available gap analysis reports in outputs/."""
    reports = []
    for path in sorted(
        OUTPUTS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    ):
        try:
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            s   = raw.get("summary", {})
            cls = s.get("classifications", {})
            reports.append({
                "filename":    path.name,
                "modified_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                "size_kb":     path.stat().st_size // 1024,
                "total":       s.get("total_regulations", s.get("total_processed", 0)),
                "coverage":    s.get("coverage_percentage", 0),
                "aligned":     cls.get("aligned",   0),
                "partial":     cls.get("partial",   0),
                "gap":         cls.get("gap",        0),
                "unmatched":   cls.get("unmatched",  0),
            })
        except Exception:
            continue
    return {"reports": reports}


@app.get("/reports/{filename}")
def get_report(filename: str):
    """Return the full content of a specific report JSON."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")
    if not filename.endswith(".json"):
        raise HTTPException(400, "Only JSON report files are supported")

    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Report '{filename}' not found")

    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(500, f"Report '{filename}' is corrupted")


@app.get("/reports/{filename}/summary")
def get_report_summary(filename: str):
    """Return only the summary block of a report."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Report '{filename}' not found")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return raw.get("summary", {})


@app.get("/reports/{filename}/gaps")
def get_gaps(
    filename:       str,
    classification: Optional[str] = None,
    risk:           Optional[str] = None,
    limit:          int = 50,
    offset:         int = 0,
):
    """
    Return filtered gap items from a report.
    Query params: ?classification=gap&risk=high&limit=20&offset=0
    """
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, f"Report '{filename}' not found")

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    items = raw.get("regulation_analysis") or raw.get("results", [])

    if classification:
        items = [i for i in items
                 if i.get("classification", "").lower() == classification.lower()]
    if risk:
        items = [i for i in items
                 if (i.get("llm_explanation") or {})
                 .get("risk_level", "").lower() == risk.lower()]

    total     = len(items)
    paginated = items[offset: offset + limit]

    return {"total": total, "offset": offset, "limit": limit, "items": paginated}


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYZE — upload PDFs and run pipeline
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze")
async def start_analysis(
    background_tasks: BackgroundTasks,
    regulation_pdf:   UploadFile = File(...),
    policy_pdf:       UploadFile = File(...),
):
    """
    Upload two PDFs and start a gap analysis job.
    Returns job_id immediately. Poll /jobs/{job_id} for status.
    """
    MAX_SIZE = 20 * 1024 * 1024  # 20 MB

    # Validate file extensions
    for upload in [regulation_pdf, policy_pdf]:
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(400, f"'{upload.filename}' is not a PDF file.")

    # Validate size and content
    reg_contents = await regulation_pdf.read()
    pol_contents = await policy_pdf.read()

    for fname, contents in [
        (regulation_pdf.filename, reg_contents),
        (policy_pdf.filename,     pol_contents),
    ]:
        if len(contents) == 0:
            raise HTTPException(400, f"'{fname}' is empty.")
        if len(contents) > MAX_SIZE:
            raise HTTPException(400, f"'{fname}' exceeds 20MB limit.")
        if not contents.startswith(b"%PDF-"):
            raise HTTPException(400, f"'{fname}' is not a valid PDF.")

    # Create job and save files
    job_id   = str(uuid.uuid4())[:8]
    reg_path = UPLOADS_DIR / f"{job_id}_regulation.pdf"
    pol_path = UPLOADS_DIR / f"{job_id}_policy.pdf"

    with open(reg_path, "wb") as f:
        f.write(reg_contents)
    with open(pol_path, "wb") as f:
        f.write(pol_contents)

    save_job({
        "job_id":     job_id,
        "status":     "queued",
        "created_at": datetime.now().isoformat(),
        "regulation": regulation_pdf.filename,
        "policy":     policy_pdf.filename,
        "step":       "",
        "result":     None,
        "error":      None,
    })

    logger.info(
        "[job:%s] Created | reg=%s | pol=%s",
        job_id, regulation_pdf.filename, policy_pdf.filename
    )

    background_tasks.add_task(_run_pipeline, job_id, reg_path, pol_path)

    return {
        "job_id":   job_id,
        "status":   "queued",
        "poll_url": f"/jobs/{job_id}",
        "message":  "Analysis started. Poll poll_url for status updates."
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE — background task
# ══════════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str, reg_path: Path, pol_path: Path):
    """
    Full production pipeline:
      1. Ingest regulation PDF -> ChromaDB regulations collection
      2. Ingest policy PDF     -> ChromaDB policies collection
      3. Run gap analysis      -> reads from ChromaDB, saves intermediate JSON
      4. Run compliance check  -> enriches with LLM, saves final report JSON
    """

    def _update(step: str, status: str = "running"):
        update_job(job_id, step=step, status=status)
        logger.info("[job:%s] %s", job_id, step)

    intermediate_path = OUTPUTS_DIR / f"api_{job_id}_gap_intermediate.json"
    final_output_path = OUTPUTS_DIR / f"api_{job_id}_report.json"

    try:
        # Step 1: Ingest regulation PDF
        _update("Ingesting regulation PDF into ChromaDB")
        from src.embeddings.ingestion_pipeline import IngestionPipeline
        embedder   = IngestionPipeline()
        reg_result = embedder.ingest_from_pipeline(
            pdf_paths       = [str(reg_path)],
            collection_name = "regulations",
            chunk_size      = 512,
            chunk_overlap   = 50
        )
        logger.info("[job:%s] Regulation ingested: %s", job_id, reg_result)

        # Step 2: Ingest policy PDF
        _update("Ingesting policy PDF into ChromaDB")
        pol_result = embedder.ingest_from_pipeline(
            pdf_paths       = [str(pol_path)],
            collection_name = "policies",
            chunk_size      = 512,
            chunk_overlap   = 50
        )
        logger.info("[job:%s] Policy ingested: %s", job_id, pol_result)

        # Step 3: Run gap analysis
        _update("Running gap analysis")
        from src.scoring.gap_analyzer import GapAnalyzer

        def _progress(percent: float):
            update_job(job_id, step=f"Gap analysis: {int(percent)}% complete")

        analyzer    = GapAnalyzer()
        gap_results = analyzer.analyze_document(
            limit             = None,
            progress_callback = _progress
        )

        if not gap_results:
            raise ValueError(
                "Gap analysis returned empty results. "
                "Check that ChromaDB has data in both collections."
            )

        with open(intermediate_path, "w", encoding="utf-8") as f:
            json.dump(gap_results, f, indent=2, ensure_ascii=False)

        logger.info(
            "[job:%s] Gap analysis complete -> %s",
            job_id, intermediate_path.name
        )

        # Step 4: Compliance verification + LLM explanations
        _update("Running compliance verification and LLM explanations")
        from run_compliance_analysis import run_compliance_analysis

        run_compliance_analysis(
            gap_report_path = str(intermediate_path),
            output_path     = str(final_output_path),
            limit           = None,
            dry_run         = False,
            use_llm         = True,
        )

        if not final_output_path.exists():
            raise ValueError(
                f"Compliance analysis produced no output at: {final_output_path}"
            )

        logger.info(
            "[job:%s] Final report saved -> %s",
            job_id, final_output_path.name
        )

        # Step 5: Clean up intermediate file
        try:
            intermediate_path.unlink()
        except Exception:
            pass

        # Done
        update_job(
            job_id,
            status = "done",
            step   = "Complete",
            result = final_output_path.name
        )
        logger.info("[job:%s] Pipeline complete", job_id)

    except ValueError as e:
        error_msg = f"Input error: {str(e)}"
        update_job(job_id, status="error", error=error_msg)
        logger.error("[job:%s] %s", job_id, error_msg)

    except FileNotFoundError as e:
        error_msg = f"File not found: {str(e)}"
        update_job(job_id, status="error", error=error_msg)
        logger.error("[job:%s] %s", job_id, error_msg)

    except ImportError as e:
        error_msg = f"Import error - check src/ modules: {str(e)}"
        update_job(job_id, status="error", error=error_msg)
        logger.error("[job:%s] %s", job_id, error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        update_job(job_id, status="error", error=error_msg)
        logger.error("[job:%s] %s", job_id, error_msg, exc_info=True)

    finally:
        for path in [reg_path, pol_path]:
            try:
                path.unlink()
                logger.debug("[job:%s] Deleted upload: %s", job_id, path.name)
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
#  JOBS — poll analysis status
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/jobs/{job_id}")
def get_job_status(job_id: str):
    """Poll analysis job status. When status=done, result has the filename."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job


@app.get("/jobs")
def list_jobs():
    """List all jobs, newest first."""
    return {"jobs": get_all_jobs()}


# ══════════════════════════════════════════════════════════════════════════════
#  EXPORT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/export/{filename}/json")
def export_json(filename: str):
    """Download a report as JSON with audit trail."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Report not found")

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    report["_audit"] = {
        "generated_at": datetime.now().isoformat(),
        "report_hash":  hashlib.md5(
            json.dumps(report, sort_keys=True).encode()
        ).hexdigest(),
        "tool": "REG-INSIGHT API v1.0",
    }
    return JSONResponse(content=report)


@app.get("/export/{filename}/markdown")
def export_markdown(filename: str):
    """Download a report as a Markdown file."""
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(400, "Invalid filename")

    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(404, "Report not found")

    try:
        from export import generate_markdown_report
    except ImportError:
        raise HTTPException(
            500,
            "week9_export.py not found. Place it in the project root next to api.py."
        )

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    md_content = generate_markdown_report(report)
    md_path    = OUTPUTS_DIR / filename.replace(".json", ".md")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return FileResponse(
        md_path,
        media_type="text/markdown",
        filename=md_path.name
    )