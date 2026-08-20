"""
Web backend for the AI Meeting Assistant.

Serves the HTML/CSS/JS front end in web/ and exposes a small JSON API over
the existing pipeline in utils/ and core/.

Run it with:
    python server.py
then open http://127.0.0.1:8000

Why a job queue instead of one blocking request: transcribing an hour-long
meeting takes minutes. An HTTP request that long would time out in the
browser and give the user no idea whether anything is happening. So a POST
starts a background job and returns immediately with an id; the page polls
that id for progress.
"""

import io
import os
import sys
import threading
import traceback
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

from utils.audio_processor import (
    chunk_audio,
    convert_to_wav,
    download_media_audio,
    save_uploaded_file,
)
from core.transcriber import transcribe_chunk
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

app = FastAPI(title="AI Meeting Assistant")

WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
TRANSCRIPT_DIR = "transcripts"
VALID_LANGUAGES = ("english", "hinglish")

# Jobs live in memory only. This is a single-user local tool, so there is no
# reason to add a database - but it does mean restarting the server loses
# them. The transcript is written to disk, so nothing expensive is lost.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _set(job_id, **fields):
    """Update a job's state. Locked because the worker thread and the HTTP
    handlers touch the same dict."""
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(fields)


def _public(job):
    """The parts of a job that are safe and useful to send to the browser.
    Skips rag_chain, which is a Python object and not serialisable."""
    return {
        "id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "progress": job["progress"],
        "error": job["error"],
        "language": job["language"],
        "source": job["source_label"],
        "created": job["created"],
        "result": job["result"],
        "chat": job["chat"],
    }


def _prepare_audio(job_id, source, uploaded_path):
    """Get a list of WAV chunks from whichever kind of source we were given."""
    if uploaded_path:
        _set(job_id, stage="Converting uploaded file", progress=8)
        wav_path = convert_to_wav(uploaded_path)
    else:
        _set(job_id, stage="Downloading audio", progress=5)
        wav_path = download_media_audio(source)

    _set(job_id, stage="Splitting audio into chunks", progress=12)
    return chunk_audio(wav_path)


def _transcribe(job_id, chunks, language):
    """
    Transcribe chunk by chunk so the UI can show real progress.

    core.transcriber.transcribe_all() does the same loop, but reports nothing
    while it runs. On a long meeting that is many silent minutes, so we drive
    the loop here instead and update the job after every chunk.
    """
    total = len(chunks)
    parts = []

    for i, chunk in enumerate(chunks):
        _set(
            job_id,
            stage=f"Transcribing chunk {i + 1} of {total}",
            # Transcription is the slow part, so it owns the widest slice of
            # the bar: 15% to 65%.
            progress=15 + int(50 * i / max(total, 1)),
        )
        parts.append(transcribe_chunk(chunk, language=language))

    return " ".join(p.strip() for p in parts if p).strip()


def _save_transcript(transcript, chunks):
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    stem = "transcript"
    if chunks:
        base = os.path.splitext(os.path.basename(chunks[0]))[0]
        stem = base.rsplit("_chunk_", 1)[0] or stem

    path = os.path.join(TRANSCRIPT_DIR, f"{stem}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)
    return path


def run_job(job_id, source, uploaded_path, language):
    """The whole pipeline, run on a background thread."""
    try:
        chunks = _prepare_audio(job_id, source, uploaded_path)
        if not chunks:
            raise RuntimeError("No audio could be extracted from that source.")

        transcript = _transcribe(job_id, chunks, language)
        if not transcript:
            raise RuntimeError(
                "Transcription produced no text. The audio may be silent or "
                "in an unsupported language."
            )

        # Written before any LLM call, so a rate limit later does not throw
        # away the expensive transcription.
        transcript_path = _save_transcript(transcript, chunks)
        _set(job_id, stage="Transcript saved", progress=66)

        _set(job_id, stage="Generating title", progress=70)
        title = generate_title(transcript)

        _set(job_id, stage="Writing summary", progress=76)
        summary = summarize(transcript)

        _set(job_id, stage="Extracting action items", progress=84)
        action_items = extract_action_items(transcript)

        _set(job_id, stage="Extracting key decisions", progress=88)
        decisions = extract_key_decisions(transcript)

        _set(job_id, stage="Extracting open questions", progress=92)
        questions = extract_questions(transcript)

        _set(job_id, stage="Building search index", progress=96)
        rag_chain = build_rag_chain(transcript)

        with JOBS_LOCK:
            JOBS[job_id]["rag_chain"] = rag_chain

        _set(
            job_id,
            status="done",
            stage="Complete",
            progress=100,
            result={
                "title": title.strip(),
                "transcript": transcript,
                "transcript_path": transcript_path,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "chunk_count": len(chunks),
            },
        )

    except Exception as err:
        # Show the user the readable message. The full traceback goes to the
        # terminal, where a developer can actually use it.
        traceback.print_exc()
        _set(job_id, status="error", stage="Failed", error=str(err))


@app.post("/api/jobs")
async def create_job(
    language: str = Form("english"),
    source_url: str = Form(""),
    file: UploadFile = File(None),
):
    language = (language or "english").strip().lower()
    if language not in VALID_LANGUAGES:
        raise HTTPException(
            400, f"Language must be one of: {', '.join(VALID_LANGUAGES)}"
        )

    source_url = (source_url or "").strip()
    if not source_url and file is None:
        raise HTTPException(400, "Provide a URL or upload a file.")

    uploaded_path = None
    if file is not None and file.filename:
        # Read it here, on the request thread, because the UploadFile stream
        # is closed once this handler returns.
        uploaded_path = save_uploaded_file(await file.read(), filename=file.filename)
        label = file.filename
    else:
        label = source_url

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {
            "id": job_id,
            "status": "running",
            "stage": "Queued",
            "progress": 0,
            "error": None,
            "language": language,
            "source_label": label,
            "created": datetime.now().isoformat(timespec="seconds"),
            "result": None,
            "rag_chain": None,
            "chat": [],
        }

    threading.Thread(
        target=run_job,
        args=(job_id, source_url, uploaded_path, language),
        daemon=True,
    ).start()

    return {"id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    with JOBS_LOCK:
        jobs = sorted(JOBS.values(), key=lambda j: j["created"], reverse=True)
        return [
            {
                "id": j["id"],
                "status": j["status"],
                "source": j["source_label"],
                "created": j["created"],
                "title": (j["result"] or {}).get("title"),
            }
            for j in jobs
        ]


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        return _public(job)


@app.post("/api/jobs/{job_id}/chat")
async def chat(job_id: str, payload: dict):
    question = (payload or {}).get("question", "").strip()
    if not question:
        raise HTTPException(400, "Question is empty.")

    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        chain = job["rag_chain"]

    if chain is None:
        raise HTTPException(409, "This meeting is still processing.")

    try:
        answer = ask_question(chain, question)
    except Exception as err:
        raise HTTPException(502, f"The model failed to answer: {err}")

    entry = {"question": question, "answer": answer}
    with JOBS_LOCK:
        JOBS[job_id]["chat"].append(entry)

    return entry


def _escape(text):
    """ReportLab parses a mini-HTML in paragraphs, so raw &, < and > break it."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _report_text(job):
    r = job["result"]
    title = r["title"]
    return "\n".join(
        [
            title,
            "=" * max(len(title), 3),
            f"Source: {job['source_label']}",
            f"Created: {job['created']}",
            "",
            "SUMMARY",
            "-------",
            r["summary"],
            "",
            "ACTION ITEMS",
            "------------",
            r["action_items"],
            "",
            "KEY DECISIONS",
            "-------------",
            r["key_decisions"],
            "",
            "OPEN QUESTIONS",
            "--------------",
            r["open_questions"],
            "",
            "FULL TRANSCRIPT",
            "---------------",
            r["transcript"],
        ]
    )


def _report_pdf(job):
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    r = job["result"]
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=r["title"],
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(_escape(r["title"]), styles["Title"]),
        Paragraph(_escape(f"Source: {job['source_label']}"), styles["Normal"]),
        Spacer(1, 12),
    ]

    for heading, body in [
        ("Summary", r["summary"]),
        ("Action Items", r["action_items"]),
        ("Key Decisions", r["key_decisions"]),
        ("Open Questions", r["open_questions"]),
        ("Full Transcript", r["transcript"]),
    ]:
        story.append(Paragraph(_escape(heading), styles["Heading2"]))
        for para in str(body).split("\n"):
            if para.strip():
                story.append(Paragraph(_escape(para), styles["BodyText"]))
        story.append(Spacer(1, 10))

    doc.build(story)
    return buffer.getvalue()


@app.get("/api/jobs/{job_id}/export")
async def export(job_id: str, format: str = "txt"):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "No such job")
        if not job["result"]:
            raise HTTPException(409, "This meeting is still processing.")

    safe = "".join(
        c for c in job["result"]["title"] if c.isalnum() or c in " -_"
    ).strip()
    safe = (safe or "meeting")[:60]

    if format == "txt":
        return Response(
            content=_report_text(job),
            media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{safe}.txt"'},
        )

    if format == "pdf":
        return Response(
            content=_report_pdf(job),
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{safe}.pdf"'},
        )

    raise HTTPException(400, "format must be txt or pdf")


@app.get("/")
async def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    print("AI Meeting Assistant -> http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
