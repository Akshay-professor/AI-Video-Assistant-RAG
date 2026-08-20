import os
import sys

from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

# Windows picks cp1252 for stdout whenever output is not a live console -
# piped to a file, captured by Streamlit, read by CI. The emoji below then
# raise UnicodeEncodeError AFTER transcription and every LLM call has already
# run, throwing away all that work over a print statement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

load_dotenv()

# Only these two are wired up. Anything else silently falls through to Whisper,
# which transcribes Hindi AS Hindi instead of translating it - a wrong result
# with no warning, which is worse than an error.
VALID_LANGUAGES = ("english", "hinglish")

TRANSCRIPT_DIR = "transcripts"


def save_transcript(transcript: str, chunks: list) -> str:
    """
    Write the transcript to disk and return its path.

    Why: everything after this point (title, summary, extractions, RAG index)
    calls an external API that can rate limit or fail. Without a saved copy,
    one failed call means re-downloading and re-transcribing the whole meeting
    - minutes of CPU for an hour-long recording. With it, retries are free.
    """
    os.makedirs(TRANSCRIPT_DIR, exist_ok=True)

    # Chunk files are named "<id>-<title>_chunk_0.wav"; strip the chunk suffix
    # so the transcript is named after the meeting it came from.
    stem = "transcript"
    if chunks:
        base = os.path.splitext(os.path.basename(chunks[0]))[0]
        stem = base.rsplit("_chunk_", 1)[0] or stem

    path = os.path.join(TRANSCRIPT_DIR, f"{stem}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return path


def run_pipeline(source: str, language: str = "english") -> dict:
    if language.lower() not in VALID_LANGUAGES:
        raise ValueError(
            f"Unknown language {language!r}. Use one of: {', '.join(VALID_LANGUAGES)}."
        )
    language = language.lower()

    print("starting AI Video Assistant")

    chunks = process_input(source)

    transcript = transcribe_all(chunks, language)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    transcript_path = save_transcript(transcript, chunks)
    print(f"Transcript saved to {transcript_path}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "transcript_path": transcript_path,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def ask_language() -> str:
    """Keep asking until the answer is one we actually support."""
    while True:
        answer = input(f"Language ({'/'.join(VALID_LANGUAGES)}): ").strip().lower()
        if not answer:
            return "english"
        if answer in VALID_LANGUAGES:
            return answer
        print(f"  '{answer}' is not supported. Choose {' or '.join(VALID_LANGUAGES)}.")


if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = ask_language()

    # process_input and the transcriber raise RuntimeError with a readable
    # message. Without this the message is buried under 40 lines of traceback.
    try:
        result = run_pipeline(source, language)
    except (RuntimeError, ValueError) as err:
        print(f"\n❌ {err}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
