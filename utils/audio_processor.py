"""
Audio input layer for the AI Meeting Assistant.

Everything that enters the app - a YouTube link, a Vimeo link, a file the user
uploaded in Streamlit, or a file already sitting on disk - comes through here
and leaves as the same thing: a list of small mono 16kHz WAV chunks that
Whisper can transcribe.

The flow diagram is at the bottom of this file.
"""

import os
import shutil

import yt_dlp
from pydub import AudioSegment

# Where every audio file we produce gets written.
#
# Why this is built from __file__ instead of just being "downloads":
# a plain relative path is resolved against the *current working directory*.
# Streamlit, pytest and "python utils/audio_processor.py" all run from
# different directories, so a relative path would scatter files in three
# different places. Anchoring to this file's location keeps them in one folder
# no matter who starts the app.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Whisper resamples whatever you give it down to mono 16kHz before doing
# anything else. So we may as well produce that shape up front:
# - the files are ~6x smaller than 48kHz stereo (a 2 hour meeting is ~230MB
#   instead of ~1.4GB)
# - pydub loads the whole file into RAM to chunk it, so smaller = less memory
TARGET_CHANNELS = 1
TARGET_SAMPLE_RATE = 16000

# Formats to try, in order, when downloading from YouTube.
#
# Why a list instead of just "bestaudio": YouTube regularly refuses individual
# format URLs with HTTP 403 while other formats on the SAME video download
# fine. Trying only the best one means a working video looks broken.
#
# The low-bitrate entries at the end are not a compromise. We downmix to mono
# 16kHz for Whisper anyway, so a 49kbps m4a carries essentially the same
# speech information as a 129kbps one - it just survives the block more often.
FORMAT_FALLBACKS = [
    "bestaudio/best",  # normal case: highest quality audio-only stream
    "140",             # m4a  129k - common, often allowed when opus is not
    "251",             # webm opus 122k
    "139",             # m4a   49k - low quality but plenty for speech
    "249",             # webm opus 50k
    "worstaudio",      # last resort: anything at all
]


def _ffmpeg_location() -> str | None:
    """
    Find the folder that holds ffmpeg.exe, or None to let yt-dlp look on PATH.

    Why this exists: ffmpeg is a separate program, not a Python package. pip
    install will never provide it. We let the user override the location with
    an env var for the case where it is installed somewhere unusual.
    """
    env_path = os.environ.get("FFMPEG_LOCATION")
    if env_path:
        return env_path

    ffmpeg = shutil.which("ffmpeg")
    return os.path.dirname(ffmpeg) if ffmpeg else None


def _require_ffmpeg() -> None:
    """
    Fail early with a readable message if ffmpeg is missing.

    Why: without this check you get a 60 line yt-dlp traceback ending in
    "ffprobe and ffmpeg not found", which buries the one thing you need to do.
    """
    if not _ffmpeg_location():
        raise RuntimeError(
            "ffmpeg not found. Install it with:  winget install Gyan.FFmpeg\n"
            "Then RESTART VS Code (a new terminal tab is not enough - the "
            "editor caches the old PATH).\n"
            "Or set FFMPEG_LOCATION to the folder containing ffmpeg.exe."
        )


def download_media_audio(url: str) -> str:
    """
    Download the audio track from a URL and return the path to a WAV file.

    Named "media" not "youtube" because yt-dlp handles ~1800 sites - Vimeo,
    SoundCloud, Twitter, Dailymotion, plus plain links to .mp3/.mp4 files.
    The same code path serves all of them.
    """
    _require_ffmpeg()

    # Filename is "<video id>-<title>.wav".
    #
    # Why the id is included: titles are not unique. Two meetings both called
    # "Weekly Standup" would overwrite each other and you would silently lose
    # a transcript. The id is stable and unique, so it also makes a good key
    # for the ChromaDB collection belonging to this meeting.
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s-%(title)s.%(ext)s")

    ydl_opts = {
        # Overwritten per attempt by the fallback loop below.
        "format": FORMAT_FALLBACKS[0],
        "outtmpl": output_template,
        "postprocessors": [
            {
                # Hands the downloaded file to ffmpeg and re-encodes to WAV.
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        # Extra flags passed straight to ffmpeg during that step.
        #   -ac 1     -> mix down to 1 channel (mono)
        #   -ar 16000 -> resample to 16kHz
        # Doing it here means ffmpeg converts once. Without this the file comes
        # out 48kHz stereo and we would need a second full pass with pydub.
        "postprocessor_args": {
            "extractaudio": ["-ac", str(TARGET_CHANNELS), "-ar", str(TARGET_SAMPLE_RATE)],
        },
        "quiet": True,
        "no_warnings": True,
        # quiet=True still lets the progress bar through, which spams the
        # Streamlit console. This turns it off properly.
        "noprogress": True,
        # YouTube hands out 403s at random on perfectly public videos - the
        # same link can fail and then succeed a second later. Without retries
        # the user just sees "download failed" and assumes the app is broken.
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 5,
        # If one player client is refused, fall back to another. YouTube
        # serves different clients from different backends, and they do not
        # fail at the same time. Order matters: default first because it has
        # the widest format coverage.
        "extractor_args": {
            "youtube": {"player_client": ["default", "tv_embedded", "web_safari"]},
        },
    }

    location = _ffmpeg_location()
    if location:
        ydl_opts["ffmpeg_location"] = location

    # Try each format in turn. YouTube blocks individual format URLs with a
    # 403 while leaving others on the SAME video perfectly downloadable, so
    # giving up after the first refusal throws away a working option.
    #
    # This cannot be done with yt-dlp's own "a/b/c" format syntax: that only
    # falls back when a format is *missing*, not when its download is refused.
    info = None
    last_error = None

    for fmt in FORMAT_FALLBACKS:
        ydl_opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except yt_dlp.utils.DownloadError as err:
            last_error = err
            message = str(err)
            # A refused or missing format is worth retrying with the next one.
            # Anything else (private video, bad link) will fail identically for
            # every format, so stop immediately rather than hammering YouTube.
            if "403" in message or "Forbidden" in message or "not available" in message:
                continue
            break

    if info is None:
        # Every format was refused. Turn yt-dlp's vaguest failures into
        # something actionable - the raw message is usually just "Forbidden",
        # which tells the user nothing.
        err = last_error
        message = str(err)

        if "DRM" in message:
            raise RuntimeError(
                "This video is DRM protected and cannot be downloaded.\n"
                "Official music videos, movie trailers and paid content are "
                "commonly protected. Try a different source, or upload a "
                "recording of the meeting instead."
            ) from err

        if "403" in message or "Forbidden" in message:
            # Two very different causes produce the same 403, so name both.
            raise RuntimeError(
                "YouTube refused the download (HTTP 403). Usual causes:\n"
                "  1. Random throttling - these 403s come and go. Just try again.\n"
                "  2. The video is region locked - try another one.\n"
                "  3. No JavaScript runtime installed (YouTube now requires one).\n"
                "     Fix:  winget install DenoLand.Deno   then restart VS Code.\n"
                "  4. yt-dlp is out of date - YouTube changes often break it.\n"
                "     Fix:  uv pip install -U yt-dlp"
            ) from err

        if "unavailable" in message.lower() or "private" in message.lower():
            raise RuntimeError(
                "That video is unavailable, private, or deleted.\n"
                "Check the link opens normally in a browser."
            ) from err

        if "age" in message.lower() and "restrict" in message.lower():
            raise RuntimeError(
                "That video is age restricted, so it cannot be downloaded "
                "without signing in. Try a different source."
            ) from err

        # Anything we did not anticipate. Still raise a clean error rather than
        # a wall of yt-dlp internals, because this surfaces in the Streamlit UI
        # where a traceback is useless to the user.
        raise RuntimeError(f"Could not download audio from that link.\n{message}") from err

    # Why not prepare_filename(): it returns the name of the file *before* the
    # WAV conversion (e.g. ".webm"). The old code string-replaced ".webm"/".m4a"
    # with ".wav" to guess the final name, which silently returned a path to a
    # deleted file whenever the format was .opus, .mp4 or .mka.
    # yt-dlp records the real, final path here - no guessing.
    downloads = info.get("requested_downloads") or []
    if downloads and downloads[0].get("filepath"):
        return downloads[0]["filepath"]

    # Safety net in case a future yt-dlp version stops filling that field.
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return os.path.splitext(ydl.prepare_filename(info))[0] + ".wav"


# Old name kept so anything already importing it keeps working.
download_youtube_audio = download_media_audio


def convert_to_wav(input_path: str) -> str:
    """
    Convert any local audio or video file to mono 16kHz WAV.

    Works on mp3, m4a, flac, ogg, opus, wav, and on video containers like mp4,
    mkv, mov, avi, webm (ffmpeg pulls the audio track out and drops the video).

    The output always lands in DOWNLOAD_DIR rather than next to the input.
    Why: with Streamlit the input often lives in a system temp folder, or on a
    read-only share. Writing beside it either litters those folders or crashes.
    """
    _require_ffmpeg()

    stem = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(DOWNLOAD_DIR, f"{stem}_converted.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(TARGET_CHANNELS).set_frame_rate(TARGET_SAMPLE_RATE)
    audio.export(output_path, format="wav")
    return output_path


def save_uploaded_file(uploaded_file, filename: str | None = None) -> str:
    """
    Write a Streamlit upload to disk and return the path.

    Why this is needed: st.file_uploader() does NOT give you a file path. It
    gives an UploadedFile object that lives in memory. ffmpeg and pydub are
    separate programs that can only read real files, so the bytes have to hit
    the disk before anything else can touch them.
    """
    # Streamlit's UploadedFile carries its original name; fall back if not.
    name = filename or getattr(uploaded_file, "name", "upload.bin")
    dest = os.path.join(DOWNLOAD_DIR, os.path.basename(name))

    # Accept raw bytes as well as a file-like object, so this also works from
    # tests or a plain script.
    data = uploaded_file if isinstance(uploaded_file, (bytes, bytearray)) else uploaded_file.read()
    with open(dest, "wb") as f:
        f.write(data)

    return dest


def chunk_audio(wav_path: str, chunk_minutes: int = 10, overlap_seconds: int = 0) -> list:
    """
    Split a long WAV into smaller pieces and return their paths.

    Why chunk at all: a 2 hour meeting held fully in memory is slow and can
    exhaust RAM. Smaller pieces also let you show progress in the UI and
    retry a single failed piece instead of the whole meeting.

    overlap_seconds repeats a few seconds at the start of each chunk. A hard
    cut at exactly 10:00 slices a word in half, which hurts the transcript and
    then poisons the RAG chunk built from it. Overlap gives Whisper the run-up
    it needs. Default is 0 so behaviour is unchanged until you opt in; 5 to 10
    seconds is a sensible value for meetings.
    """
    audio = AudioSegment.from_file(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000
    overlap_ms = overlap_seconds * 1000

    # Chunks go to DOWNLOAD_DIR, and the ".wav" is stripped from the stem first.
    # The old code produced names like "meeting.wav_chunk_0.wav".
    stem = os.path.splitext(os.path.basename(wav_path))[0]

    chunks = []
    step = chunk_ms - overlap_ms  # how far we advance each time
    for i, start in enumerate(range(0, len(audio), step)):
        # Rewind by the overlap, except for the very first chunk.
        piece_start = max(0, start - overlap_ms) if i else 0
        piece = audio[piece_start:start + chunk_ms]

        # A tiny trailing sliver adds nothing but a wasted Whisper call.
        if len(piece) < 1000:
            continue

        chunk_path = os.path.join(DOWNLOAD_DIR, f"{stem}_chunk_{i}.wav")
        piece.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source, chunk_minutes: int = 10, overlap_seconds: int = 0) -> list:
    """
    The single entry point. Give it anything; get back a list of WAV chunks.

    "source" can be:
      - a URL string          -> downloaded with yt-dlp
      - a local path string   -> converted with pydub
      - a Streamlit upload    -> saved to disk first, then converted
    """
    # An upload has no path, so it is handled first: get it onto disk, then it
    # is just an ordinary local file from here on.
    if not isinstance(source, str):
        print("Detected uploaded file. Saving to disk...")
        source = save_uploaded_file(source)
        wav_path = convert_to_wav(source)

    elif source.startswith(("http://", "https://")):
        print("Detected URL. Downloading audio...")
        wav_path = download_media_audio(source)

    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"No such file: {source}")
        print("Detected local file. Converting to WAV...")
        wav_path = convert_to_wav(source)

    print("Chunking audio...")
    chunks = chunk_audio(wav_path, chunk_minutes, overlap_seconds)
    print(f"Audio ready - {len(chunks)} chunk(s) created.")
    return chunks


# ---------------------------------------------------------------------------
# THE FLOW - what happens when a source shows up
# ---------------------------------------------------------------------------
#
#                        user gives us something
#                                 |
#                          process_input(source)
#                                 |
#            +--------------------+--------------------+
#            |                    |                    |
#      not a string         starts with            anything else
#      (Streamlit           http:// or             (a path on disk)
#       upload)             https://                     |
#            |                    |                      |
#   save_uploaded_file()   download_media_audio()        |
#   bytes -> real file     yt-dlp grabs the audio        |
#            |             stream only, ffmpeg           |
#            |             converts to mono 16k          |
#            |                    |                      |
#            +---------> convert_to_wav() <--------------+
#                        ffmpeg reads any format
#                        (mp3/mp4/mkv/flac/...)
#                        and writes mono 16kHz WAV
#                                 |
#                        one clean WAV in downloads/
#                                 |
#                          chunk_audio()
#                        cut into ~10 min pieces
#                                 |
#                     [chunk_0.wav, chunk_1.wav, ...]
#                                 |
#                                 v
#              Whisper transcribes each chunk in order
#                                 |
#              Hindi? -> translate to English
#                                 |
#              full transcript text
#                     /                    \
#         LangChain LCEL + Mistral      ChromaDB + HF embeddings
#         summary, action items,        so the user can chat
#         key decisions                 with the transcript
#                     \                    /
#                       export as PDF / TXT
#
# The point of the three-way split at the top: every branch ends at the same
# place, a single mono 16kHz WAV. Everything downstream - Whisper, the summary
# chain, the RAG index - only ever sees that one shape and never has to care
# whether the meeting came from YouTube or a phone recording.
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    # Quick manual test.
    #
    #   python utils/audio_processor.py "https://youtu.be/SOME_ID"
    #   python utils/audio_processor.py "C:\path\to\meeting.mp3"
    #
    # Pass the link/path as an argument so you don't have to edit this file
    # every time. With no argument it falls back to a short public video that
    # is known to work, which is handy for checking the setup itself.
    import sys

    source = sys.argv[1] if len(sys.argv) > 1 else "https://youtu.be/aM5boWsh7yI?si=9BFZikiTm65z1rFC"

    for path in process_input(source):
        print(path)
