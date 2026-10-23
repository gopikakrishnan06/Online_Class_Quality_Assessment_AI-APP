
from pathlib import Path
from urllib.parse import urlparse
import hashlib
import ipaddress
import json
import re
import socket
import subprocess
import tempfile
import urllib.request

import cv2
import imageio_ffmpeg
import numpy as np
import pandas as pd
import plotly.express as px
import soundfile as sf
import streamlit as st
from faster_whisper import WhisperModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Classroom Quality Assessment",
    page_icon="🎓",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = Path(tempfile.gettempdir()) / "class_quality_assessment"
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Category weights must total 100.
WEIGHTS = {
    "Audio Quality": 10,
    "Visual Quality": 10,
    "Content Quality": 25,
    "Teaching Style": 20,
    "Engagement Indicators": 20,
    "Communication": 15,
}

SUPPORTED_VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".mpeg", ".mpg"
}


# ============================================================
# 2. SESSION STATE
# ============================================================

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "analysis_fingerprint" not in st.session_state:
    st.session_state["analysis_fingerprint"] = None

if "prepared_url_fingerprint" not in st.session_state:
    st.session_state["prepared_url_fingerprint"] = None

if "prepared_url_path" not in st.session_state:
    st.session_state["prepared_url_path"] = None


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def file_hash(data: bytes) -> str:
    """Create a stable fingerprint for uploaded video bytes."""
    return hashlib.sha256(data).hexdigest()


def safe_filename(name: str) -> str:
    """Remove unsafe characters from a filename."""
    name = Path(name).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    return name or "uploaded_video.mp4"


def run_command(command: list[str]) -> None:
    """Run an external command and report errors."""
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        error_text = result.stderr[-2500:] if result.stderr else (
            "No error details were returned."
        )
        raise RuntimeError(error_text)


def validate_public_url(url: str) -> None:
    """Basic validation for a public, direct HTTP(S) video URL."""
    parsed = urlparse(url.strip())

    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Use a URL beginning with http:// or https://.")

    if not parsed.hostname:
        raise ValueError("The URL does not contain a valid hostname.")

    if parsed.username or parsed.password:
        raise ValueError("URLs containing embedded credentials are not supported.")

    hostname = parsed.hostname

    try:
        addresses = [ipaddress.ip_address(hostname)]
    except ValueError:
        try:
            addresses = [
                ipaddress.ip_address(item[4][0])
                for item in socket.getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
            ]
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve the URL hostname: {exc}") from exc

    if not addresses or any(not address.is_global for address in addresses):
        raise ValueError("The URL must point to a publicly accessible host.")


def download_video(url: str, destination: Path) -> Path:
    """
    Download a direct video file URL.
    This does not download videos from YouTube or other streaming pages.
    """
    validate_public_url(url)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ClassQualityAssessment/1.0"},
    )

    with urllib.request.urlopen(request, timeout=60) as response:
        final_url = response.geturl()
        validate_public_url(final_url)

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" in content_type:
            raise ValueError(
                "This URL returned a web page, not a direct video file. "
                "Use a direct downloadable video URL."
            )

        total_size = 0
        max_size = 1024 * 1024 * 1024  # 1 GB limit for this prototype

        with destination.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                total_size += len(chunk)

                if total_size > max_size:
                    raise ValueError(
                        "Video exceeds the 1 GB prototype download limit."
                    )

                output.write(chunk)

    if total_size == 0:
        raise ValueError("The downloaded video file is empty.")

    if not destination.exists():
        raise FileNotFoundError("The downloaded video could not be saved.")

    return destination


def prepare_uploaded_video(uploaded_file) -> tuple[Path, str]:
    """Save an uploaded video to a stable temporary path."""
    video_bytes = uploaded_file.getvalue()
    fingerprint = file_hash(video_bytes)

    extension = Path(uploaded_file.name).suffix.lower()

    if extension not in SUPPORTED_VIDEO_EXTENSIONS:
        raise ValueError(
            "Unsupported video format. Upload MP4, MOV, AVI, MKV, or WEBM."
        )

    destination = WORK_DIR / f"{fingerprint}{extension}"

    if not destination.exists():
        destination.write_bytes(video_bytes)

    return destination, fingerprint




# ============================================================
# CATEGORY CLASSIFICATION
# ============================================================

def classify_score(score):
    """Convert a numerical score into a provisional category."""

    if score is None:
        return "Not Available"

    if not np.isfinite(score):
        return "Not Available"

    if score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"



# ============================================================
# 4. AUDIO EXTRACTION AND ANALYSIS
# ============================================================

def extract_audio(video_path: Path, wav_path: Path) -> None:
    """Extract mono 16 kHz PCM audio from the actual video."""
    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()

    command = [
        ffmpeg_path,
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(wav_path),
    ]

    run_command(command)

    if not wav_path.exists() or wav_path.stat().st_size == 0:
        raise RuntimeError(
            "Could not extract audio. Check whether the video contains an audio track."
        )


def analyze_audio(wav_path: Path) -> dict:
    """Calculate measurable acoustic features from the extracted audio."""
    audio, sample_rate = sf.read(wav_path, dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if len(audio) == 0:
        raise ValueError("The extracted audio is empty.")

    audio = np.nan_to_num(audio)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))

    # Divide the waveform into 20 ms frames.
    frame_size = max(1, int(sample_rate * 0.02))
    frame_rms = []

    for start in range(0, len(audio), frame_size):
        frame = audio[start:start + frame_size]

        if len(frame):
            frame_rms.append(float(np.sqrt(np.mean(frame ** 2))))

    frame_rms = np.asarray(frame_rms, dtype=float)

    # This is an operational silence threshold, not a universal definition
    # of silence.
    silence_threshold = 0.01
    silence_percentage = float(
        np.mean(frame_rms < silence_threshold) * 100
    )

    clipping_percentage = float(
        np.mean(np.abs(audio) >= 0.999) * 100
    )

    duration_seconds = len(audio) / sample_rate

    # Provisional technical quality score.
    # Silence can represent legitimate pauses, so it is not treated as
    # proof of poor teaching.
    level_score = min(100.0, rms / 0.08 * 100.0)
    clipping_penalty = min(40.0, clipping_percentage * 10.0)

    audio_score = float(
        np.clip(level_score - clipping_penalty, 0, 100)
    )

    return {
        "sample_rate": int(sample_rate),
        "duration_seconds": float(duration_seconds),
        "mean_rms_energy": rms,
        "peak_amplitude": peak,
        "silence_percentage": silence_percentage,
        "clipping_percentage": clipping_percentage,
        "audio_score": audio_score,
    }


# ============================================================
# 5. VISUAL ANALYSIS
# ============================================================

def analyze_video_frames(video_path: Path, max_frames: int = 60) -> dict:
    """
    Sample frames across the recording and calculate basic image features.
    These features do not identify teaching behaviour or inclusiveness.
    """
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise ValueError("OpenCV could not open this video.")

    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration_seconds = (
        frame_count / fps if fps and fps > 0 else 0.0
    )

    if frame_count <= 0:
        capture.release()
        raise ValueError("The video contains no readable frames.")

    sample_count = min(max_frames, frame_count)
    sample_indices = np.linspace(
        0,
        frame_count - 1,
        num=sample_count,
        dtype=int,
    )

    brightness_values = []
    contrast_values = []
    sharpness_values = []
    processed_frames = 0

    for frame_index in sample_indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = capture.read()

        if not success or frame is None:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness_values.append(float(np.mean(gray)))
        contrast_values.append(float(np.std(gray)))
        sharpness_values.append(
            float(cv2.Laplacian(gray, cv2.CV_64F).var())
        )

        processed_frames += 1

    capture.release()

    if processed_frames == 0:
        raise ValueError("Could not read any video frames.")

    mean_brightness = float(np.mean(brightness_values))
    mean_contrast = float(np.mean(contrast_values))
    mean_sharpness = float(np.mean(sharpness_values))

    # Simple technical visibility heuristic, not a teaching-quality model.
    brightness_score = max(
        0.0,
        100.0 - abs(mean_brightness - 127.5) / 127.5 * 100.0,
    )

    contrast_score = min(100.0, mean_contrast / 60.0 * 100.0)
    sharpness_score = min(100.0, mean_sharpness / 300.0 * 100.0)

    visual_score = float(
        np.mean([
            brightness_score,
            contrast_score,
            sharpness_score,
        ])
    )

    return {
        "width": width,
        "height": height,
        "fps": float(fps) if fps and fps > 0 else 0.0,
        "frame_count": frame_count,
        "duration_seconds": float(duration_seconds),
        "sampled_frames": processed_frames,
        "mean_brightness": mean_brightness,
        "mean_contrast": mean_contrast,
        "mean_sharpness": mean_sharpness,
        "visual_score": visual_score,
    }


# ============================================================
# 6. SPEECH TRANSCRIPTION
# ============================================================

@st.cache_resource(show_spinner="Loading the speech recognition model...")
def load_whisper_model():
    """Load the speech recognition model once per Streamlit session."""
    return WhisperModel(
        "tiny",
        device="cpu",
        compute_type="int8",
    )


def transcribe_audio(wav_path: Path) -> dict:
    """Transcribe actual extracted speech using Faster Whisper."""
    model = load_whisper_model()

    segments, info = model.transcribe(
        str(wav_path),
        beam_size=3,
        vad_filter=True,
    )

    transcript_segments = []
    text_parts = []

    for segment in segments:
        text = segment.text.strip()

        if text:
            text_parts.append(text)
            transcript_segments.append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            })

    transcript = " ".join(text_parts).strip()

    return {
        "transcript": transcript,
        "segments": transcript_segments,
        "detected_language": getattr(info, "language", "unknown"),
        "language_probability": float(
            getattr(info, "language_probability", 0.0)
        ),
        "speech_segment_count": len(transcript_segments),
    }


# ============================================================
# 7. TRANSCRIPT-BASED INDICATORS
# ============================================================

def count_matches(text: str, patterns: list[str]) -> int:
    """Count simple word/phrase matches."""
    return sum(
        len(re.findall(pattern, text, flags=re.IGNORECASE))
        for pattern in patterns
    )


def calculate_transcript_scores(
    transcript: str,
    audio_metrics: dict,
    video_metrics: dict,
) -> dict:
    """
    Transparent rule-based indicators.
    These are not validated models of teaching effectiveness.
    """
    text = transcript.lower().strip()
    words = re.findall(r"\b[\w'-]+\b", text)
    word_count = len(words)

    duration_minutes = max(
        audio_metrics["duration_seconds"] / 60.0,
        1 / 60.0,
    )

    words_per_minute = word_count / duration_minutes

    explanation_patterns = [
        r"\bfor example\b",
        r"\bfor instance\b",
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bthat means\b",
        r"\bin other words\b",
        r"\blet us\b",
        r"\blet's\b",
    ]

    question_patterns = [
        r"\?",
        r"\bwhy\b",
        r"\bhow\b",
        r"\bwhat do you think\b",
        r"\bcan anyone\b",
        r"\bany questions\b",
        r"\bdoes that make sense\b",
    ]

    inclusive_patterns = [
        r"\beveryone\b",
        r"\ball of you\b",
        r"\banyone\b",
        r"\bplease share\b",
        r"\bfeel free\b",
        r"\blet us hear\b",
        r"\byour ideas\b",
        r"\bany questions\b",
    ]

    structure_patterns = [
        r"\btoday we will\b",
        r"\blearning objective\b",
        r"\bfirst\b",
        r"\bsecond\b",
        r"\bfinally\b",
        r"\bto summarise\b",
        r"\bto summarize\b",
        r"\bin conclusion\b",
        r"\brecap\b",
    ]

    encouragement_patterns = [
        r"\bgood question\b",
        r"\bwell done\b",
        r"\bexcellent\b",
        r"\bthat's right\b",
        r"\bthank you for sharing\b",
        r"\bgreat effort\b",
    ]

    explanation_count = count_matches(text, explanation_patterns)
    question_count = count_matches(text, question_patterns)
    inclusive_count = count_matches(text, inclusive_patterns)
    structure_count = count_matches(text, structure_patterns)
    encouragement_count = count_matches(text, encouragement_patterns)

    # Scores are intentionally capped and transparent.
    content_score = float(np.clip(
        30
        + min(explanation_count, 8) * 7
        + min(structure_count, 5) * 6
        + min(word_count / 100, 5) * 3,
        0,
        100,
    ))

    teaching_style_score = float(np.clip(
        35
        + min(explanation_count, 5) * 7
        + min(structure_count, 4) * 6
        + min(encouragement_count, 4) * 5,
        0,
        100,
    ))

    engagement_score = float(np.clip(
        30
        + min(question_count, 8) * 8
        + min(inclusive_count, 5) * 5
        + min(encouragement_count, 4) * 4,
        0,
        100,
    ))

    # A broad speaking-rate indicator. A rate outside this range does not
    # necessarily indicate poor communication.
    if 90 <= words_per_minute <= 170:
        rate_score = 100.0
    elif words_per_minute < 90:
        rate_score = max(30.0, words_per_minute / 90.0 * 100.0)
    else:
        rate_score = max(
            30.0,
            100.0 - (words_per_minute - 170.0) * 0.8,
        )

    communication_score = float(np.clip(
        0.75 * rate_score
        + 0.25 * audio_metrics["audio_score"],
        0,
        100,
    ))

    return {
        "word_count": word_count,
        "words_per_minute": float(words_per_minute),
        "explanation_indicators": explanation_count,
        "question_indicators": question_count,
        "inclusive_language_indicators": inclusive_count,
        "structure_indicators": structure_count,
        "encouragement_indicators": encouragement_count,
        "content_score": content_score,
        "teaching_style_score": teaching_style_score,
        "engagement_score": engagement_score,
        "communication_score": communication_score,
    }


# ============================================================
# 8. COMPLETE RECORDING ASSESSMENT
# ============================================================

def analyze_recording(video_path: Path, source_name: str) -> dict:
    """Run the actual audio, video, and transcript pipeline."""
    if not video_path.exists():
        raise FileNotFoundError(
            "The source video is no longer available. Please select it again."
        )

    wav_path = WORK_DIR / f"{video_path.stem}_audio.wav"

    progress = st.progress(0, text="Preparing recording...")

    try:
        progress.progress(10, text="Extracting audio...")
        extract_audio(video_path, wav_path)

        progress.progress(25, text="Analyzing audio features...")
        audio_metrics = analyze_audio(wav_path)

        progress.progress(45, text="Analyzing sampled video frames...")
        video_metrics = analyze_video_frames(video_path)

        progress.progress(65, text="Transcribing speech...")
        speech_metrics = transcribe_audio(wav_path)

        progress.progress(85, text="Calculating assessment indicators...")
        transcript_scores = calculate_transcript_scores(
            speech_metrics["transcript"],
            audio_metrics,
            video_metrics,
        )

        category_scores = {
            "Audio Quality": audio_metrics["audio_score"],
            "Visual Quality": video_metrics["visual_score"],
            "Content Quality": transcript_scores["content_score"],
            "Teaching Style": transcript_scores["teaching_style_score"],
            "Engagement Indicators": transcript_scores["engagement_score"],
            "Communication": transcript_scores["communication_score"],
        }

        weighted_score = sum(
            category_scores[category] * weight
            for category, weight in WEIGHTS.items()
        ) / 100.0

       

        score_table = pd.DataFrame([
    {
        "Category": category,
        "Score": round(float(category_scores[category]), 2),
        "Rating": classify_score(category_scores[category]),
        "Weight (%)": weight,
        "Weighted Contribution": round(
            float(category_scores[category]) * weight / 100.0,
            2,
        ),
    }
    for category, weight in WEIGHTS.items()
])



        progress.progress(100, text="Assessment completed.")

        return {
            "source_name": source_name,
            "overall_score": round(float(weighted_score), 2),
            "category_scores": category_scores,
            "score_table": score_table,
            "audio": audio_metrics,
            "video": video_metrics,
            "speech": speech_metrics,
            "transcript_scores": transcript_scores,
            "weights": WEIGHTS,
            "assessment_note": (
                "Prototype heuristic assessment. Scores are not validated "
                "measures of teaching effectiveness, inclusiveness, or "
                "student engagement."
            ),
        }

    finally:
        progress.empty()


# ============================================================
# 9. REPORT DISPLAY
# ============================================================


# ============================================================
# 
# ============================================================
# 9. DISPLAY ASSESSMENT REPORT
# ============================================================

def show_report(result: dict) -> None:
    """Display the assessment report for the current source."""

    st.divider()
    st.header("📊 Classroom Quality Assessment Report")

    st.caption(f"Analyzed source: {result['source_name']}")

    st.warning(result["assessment_note"])

    # --------------------------------------------------------
    # 9.1 Overall Assessment Metrics
    # --------------------------------------------------------

    overall = float(result["overall_score"])
    overall_category = classify_score(overall)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Overall Weighted Score",
        f"{overall:.1f} / 100",
    )

    col2.metric(
        "Overall Category",
        overall_category,
    )

    col3.metric(
        "Recording Duration",
        f"{result['audio']['duration_seconds'] / 60:.1f} min",
    )

    col4.metric(
        "Transcript Word Count",
        result["transcript_scores"]["word_count"],
    )

    # --------------------------------------------------------
    # 9.2 Category Scores AND CATEGORY NAMES
    # --------------------------------------------------------

    st.subheader("Category Scores and Ratings")

    # Copy the existing score table
    score_table = result["score_table"].copy()

    # Make sure scores are numeric
    score_table["Score"] = pd.to_numeric(
        score_table["Score"],
        errors="coerce",
    )

    # ADD THE CATEGORY NAME FOR EVERY DIMENSION
    score_table["Rating"] = score_table["Score"].apply(
        lambda score: (
            classify_score(float(score))
            if pd.notna(score)
            else "Not Available"
        )
    )

    # Display chart
    fig = px.bar(
        score_table,
        x="Category",
        y="Score",
        color="Rating",
        range_y=[0, 100],
        text="Score",
        hover_data=["Rating"],
        title="Category Scores and Ratings",
    )

    fig.update_layout(
        xaxis_tickangle=-25,
        yaxis_title="Score (0–100)",
        xaxis_title="Assessment Category",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # Display score AND rating name in the table
    display_columns = [
        column
        for column in [
            "Category",
            "Score",
            "Rating",
            "Weight (%)",
            "Weighted Contribution",
        ]
        if column in score_table.columns
    ]

    st.dataframe(
        score_table[display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # 9.3 Audio Quality Analysis
    # --------------------------------------------------------

    st.subheader("Audio Quality Analysis")

    audio_metrics = result["audio"]

    audio_col1, audio_col2, audio_col3 = st.columns(3)

    audio_col1.metric(
        "Audio Duration",
        f"{audio_metrics['duration_seconds'] / 60:.1f} min",
    )

    audio_col2.metric(
        "Sample Rate",
        f"{audio_metrics.get('sample_rate', 0):,} Hz",
    )

    audio_score = result["category_scores"].get(
        "Audio Quality"
    )

    audio_col3.metric(
        "Audio Quality Rating",
        classify_score(float(audio_score))
        if audio_score is not None
        else "Not Available",
    )

    with st.expander("View audio analysis details"):
        st.json(audio_metrics)

    # --------------------------------------------------------
    # 9.4 Visual Quality Analysis
    # --------------------------------------------------------

    st.subheader("Visual Quality Analysis")

    video_metrics = result["video"]

    visual_col1, visual_col2 = st.columns(2)

    visual_score = result["category_scores"].get(
        "Visual Quality"
    )

    visual_col1.metric(
        "Visual Quality Score",
        f"{float(visual_score):.1f} / 100"
        if visual_score is not None
        else "Not Available",
    )

    visual_col2.metric(
        "Visual Quality Rating",
        classify_score(float(visual_score))
        if visual_score is not None
        else "Not Available",
    )

    with st.expander("View visual analysis details"):
        st.json(video_metrics)

    # --------------------------------------------------------
    # 9.5 Speech Analysis
    # --------------------------------------------------------

    st.subheader("Speech Analysis")

    speech_metrics = result["speech"]

    with st.expander("View speech analysis details"):
        st.json(speech_metrics)

    # --------------------------------------------------------
    # 9.6 Transcript and Teaching Indicators
    # --------------------------------------------------------

    st.subheader("Transcript and Teaching Indicators")

    transcript_scores = result["transcript_scores"]

    st.json(transcript_scores)

    # --------------------------------------------------------
    # 9.7 Download Assessment Results
    # --------------------------------------------------------

    st.subheader("Download Assessment Results")

    import json

    col_json, col_csv = st.columns(2)

    with col_json:
        json_data = json.dumps(
            result,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

        st.download_button(
            label="Download Full Report (JSON)",
            data=json_data,
            file_name="class_quality_assessment.json",
            mime="application/json",
        )

    with col_csv:
        csv_data = score_table.to_csv(index=False)

        st.download_button(
            label="Download Category Scores (CSV)",
            data=csv_data,
            file_name="class_quality_scores.csv",
            mime="text/csv",
        )

    # --------------------------------------------------------
    # 9.8 Assessment Disclaimer
    # --------------------------------------------------------

    st.caption(
        "This is a prototype heuristic assessment. "
        "The scores and categories are not validated measures "
        "of teaching effectiveness, inclusiveness, or student "
        "engagement."
    )



# ============================================================
# 10. STREAMLIT USER INTERFACE
# ============================================================

st.title("🎓 Classroom Quality Assessment")
st.write(
    "Analyze a classroom recording using audio features, sampled video "
    "frames, speech transcription, and transparent prototype indicators."
)

st.caption(
    "Choose a video source, then press the assessment button. "
    "No assessment report is generated until you press the button."
)

source_mode = st.radio(
    "Select video source",
    ["Upload Video", "Direct Video URL"],
    horizontal=True,
)

video_path = None
source_fingerprint = None
source_name = None
source_error = None


# ------------------------------------------------------------
# 10A. Upload mode
# ------------------------------------------------------------

if source_mode == "Upload Video":
    uploaded_file = st.file_uploader(
        "Upload a classroom recording",
        type=["mp4", "mov", "avi", "mkv", "webm", "mpeg", "mpg"],
        help="Upload a video containing the classroom recording.",
    )

    if uploaded_file is not None:
        try:
            video_path, source_fingerprint = prepare_uploaded_video(
                uploaded_file
            )
            source_name = safe_filename(uploaded_file.name)

        except Exception as exc:
            source_error = str(exc)


# ------------------------------------------------------------
# 10B. URL mode
# ------------------------------------------------------------

else:
    video_url = st.text_input(
        "Direct downloadable video URL",
        placeholder="https://example.com/classroom-recording.mp4",
        help=(
            "Use a direct URL to a video file. Ordinary YouTube watch pages "
            "and streaming-page URLs are not supported."
        ),
    ).strip()

    if video_url:
        source_fingerprint = hashlib.sha256(
            video_url.encode("utf-8")
        ).hexdigest()

        source_name = "Video URL"

        cached_path_string = st.session_state.get("prepared_url_path")
        cached_fingerprint = st.session_state.get(
            "prepared_url_fingerprint"
        )

        if (
            cached_fingerprint == source_fingerprint
            and cached_path_string
            and Path(cached_path_string).exists()
        ):
            video_path = Path(cached_path_string)

        else:
            # Download the source once for this URL. Analysis still requires
            # the button below.
            if st.button(
                "Prepare Video URL",
                use_container_width=True,
            ):
                try:
                    with st.spinner("Downloading the direct video file..."):
                        destination = WORK_DIR / (
                            f"{source_fingerprint}.video"
                        )

                        video_path = download_video(
                            video_url,
                            destination,
                        )

                        st.session_state["prepared_url_path"] = str(
                            video_path
                        )
                        st.session_state["prepared_url_fingerprint"] = (
                            source_fingerprint
                        )

                    st.success(
                        "Video prepared. Press Assess Classroom Quality "
                        "to analyze it."
                    )
                    st.rerun()

                except Exception as exc:
                    source_error = str(exc)


# ------------------------------------------------------------
# 10C. Clear stale reports when input changes
# ------------------------------------------------------------

saved_fingerprint = st.session_state.get("analysis_fingerprint")

if (
    saved_fingerprint is not None
    and saved_fingerprint != source_fingerprint
):
    st.session_state["analysis_result"] = None
    st.session_state["analysis_fingerprint"] = None


if source_error:
    st.error(f"Could not prepare the video: {source_error}")


# ------------------------------------------------------------
# 10D. Assessment button
# ------------------------------------------------------------

if video_path is not None and source_fingerprint is not None:
    st.success("Video is ready for assessment.")

    if st.button(
        "🎯 Assess Classroom Quality",
        type="primary",
        use_container_width=True,
    ):
        # Clear the previous result before starting the new assessment.
        st.session_state["analysis_result"] = None
        st.session_state["analysis_fingerprint"] = None

        try:
            with st.spinner(
                "Analyzing the selected recording. This may take a while..."
            ):
                result = analyze_recording(
                    video_path,
                    source_name or video_path.name,
                )

            st.session_state["analysis_result"] = result
            st.session_state["analysis_fingerprint"] = source_fingerprint

            st.success("Assessment completed successfully!")

        except Exception as exc:
            st.error(f"Assessment failed: {exc}")


# ------------------------------------------------------------
# 10E. Display ONLY the current input's result
# ------------------------------------------------------------

if (
    source_fingerprint is not None
    and st.session_state.get("analysis_fingerprint")
        == source_fingerprint
    and st.session_state.get("analysis_result") is not None
):
    show_report(st.session_state["analysis_result"])

elif video_path is None:
    st.info(
        "Upload a video, or enter a direct video URL and prepare it. "
        "Then press the assessment button to generate a report."
    )


