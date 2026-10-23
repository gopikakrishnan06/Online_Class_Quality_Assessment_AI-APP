
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# 1. PATHS AND DIMENSION WEIGHTS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_DIR / "evaluation"

AUDIO_CSV = EVALUATION_DIR / "audio_features.csv"
VIDEO_CSV = EVALUATION_DIR / "visual_video_summary.csv"
TRANSCRIPT_CSV = EVALUATION_DIR / "transcript_features.csv"

WEIGHTS = {
    "audio_quality": 10,
    "visual_quality": 10,
    "content_quality": 25,
    "teaching_style": 20,
    "engagement": 20,
    "communication": 15,
}


# --------------------------------------------------
# 2. GENERAL HELPER FUNCTIONS
# --------------------------------------------------

def safe_number(row, column, default=None):
    """Return a numeric feature, or default if unavailable."""
    value = row.get(column, default)

    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value, minimum=0, maximum=100):
    """Keep a score between 0 and 100."""
    return max(minimum, min(maximum, value))


def range_score(value, low, high):
    """
    Provisional score:
    values inside the acceptable range get 100;
    values outside the range lose points progressively.
    """
    if value is None:
        return None

    if low <= value <= high:
        return 100.0

    if value < low:
        return clamp(100 * value / low) if low > 0 else 0.0

    # Above the preferred range
    return clamp(100 * high / value) if value > 0 else 0.0


def average_available(scores):
    """Average only available component scores."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 2)


def normalize_recording_name(filename):
    """Normalize audio/video recording filenames for matching."""
    name = Path(str(filename)).stem.strip().casefold()

    if name.endswith("_audio"):
        name = name[:-6]

    return name.strip()


def normalize_transcript_name(filename):
    """Remove common transcript suffixes."""
    name = Path(str(filename)).stem.strip().casefold()

    suffixes = (
        "-transcript-1.1 (1)",
        "-transcript-1.1",
        "-transcript",
    )

    for suffix in suffixes:
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            break

    return name


# --------------------------------------------------
# 3. MATCH AUDIO, VIDEO AND TRANSCRIPT DATA
# --------------------------------------------------

def load_and_match_data():
    for path in (AUDIO_CSV, VIDEO_CSV, TRANSCRIPT_CSV):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    audio_df = pd.read_csv(AUDIO_CSV)
    video_df = pd.read_csv(VIDEO_CSV)
    transcript_df = pd.read_csv(TRANSCRIPT_CSV)

    audio_df["_key"] = audio_df["file_name"].apply(
        normalize_recording_name
    )
    video_df["_key"] = video_df["video_name"].apply(
        normalize_recording_name
    )

    # Merge only when the normalized names match.
    combined = audio_df.merge(
        video_df,
        on="_key",
        how="left",
        suffixes=("_audio", "_video"),
        indicator=True,
    )

    combined["match_status"] = combined["_merge"].map({
        "both": "Matched",
        "left_only": "Video not found",
        "right_only": "Video only",
    })

    transcript_df["_transcript_key"] = (
        transcript_df["transcript_file"].apply(
            normalize_transcript_name
        )
    )

    transcript_rows = []

    for _, recording in combined.iterrows():
        recording_key = recording["_key"]

        # Remove the dated suffix, e.g. " - 2026-07-25".
        identity = recording_key.rsplit(" - ", 1)[0]

        category = str(recording.get("category", "")).strip().casefold()

        candidates = transcript_df[
            transcript_df["category"]
            .astype(str)
            .str.strip()
            .str.casefold()
            .eq(category)
        ].copy()

        # Transcript filenames don't contain recording dates.
        # This is a candidate match, not proof of exact identity.
        candidates = candidates[
            candidates["_transcript_key"].apply(
                lambda name: (
                    name == identity
                    or name.startswith(identity + " ")
                    or identity.startswith(name + " ")
                )
            )
        ]

        if len(candidates) == 1:
            transcript_rows.append(candidates.iloc[0].to_dict())
        else:
            transcript_rows.append({})

    transcript_features = pd.DataFrame(transcript_rows).reset_index(drop=True)

    # Keep one row per recording; attach transcript feature columns.
    transcript_features = transcript_features.drop(
        columns=["category", "transcript_file", "_transcript_key"],
        errors="ignore",
    )

    combined = combined.reset_index(drop=True)

    for column in transcript_features.columns:
        combined[column] = transcript_features[column]

    return combined


# --------------------------------------------------
# 4. AUDIO QUALITY SCORE
# --------------------------------------------------

def score_audio(row):
    """
    Heuristic estimate based on audio level, silence and clipping.
    Silence can include legitimate pauses, so its penalty is limited.
    """
    rms = safe_number(row, "mean_rms_energy")
    silence = safe_number(row, "silence_percentage")
    clipping = safe_number(row, "clipping_percentage")

    components = []

    if rms is not None:
        # Provisional range; recording gain affects RMS.
        components.append(range_score(rms, 0.01, 0.10))

    if silence is not None:
        # Do not treat every pause as a defect.
        silence_score = 100 - max(0, silence - 50) * 0.8
        components.append(clamp(silence_score))

    if clipping is not None:
        clipping_score = 100 - clipping * 10
        components.append(clamp(clipping_score))

    return average_available(components)


# --------------------------------------------------
# 5. VISUAL QUALITY SCORE
# --------------------------------------------------

def score_visual(row):
    brightness = safe_number(row, "mean_brightness")
    contrast = safe_number(row, "mean_contrast")
    sharpness = safe_number(row, "mean_sharpness")
    dark_frames = safe_number(row, "dark_frame_percentage")
    overexposed = safe_number(
        row, "potential_overexposed_frame_percentage"
    )

    components = []

    if brightness is not None:
        # Provisional preferred brightness range.
        components.append(range_score(brightness, 50, 180))

    if contrast is not None:
        components.append(range_score(contrast, 20, 100))

    if sharpness is not None:
        components.append(range_score(sharpness, 50, 500))

    if dark_frames is not None:
        components.append(
            clamp(100 - max(0, dark_frames - 10) * 1.5)
        )

    if overexposed is not None:
        components.append(
            clamp(100 - overexposed * 1.5)
        )

    return average_available(components)


# --------------------------------------------------
# 6. CONTENT QUALITY SCORE
# --------------------------------------------------

def score_content(row):
    """
    Transcript indicator rates are proxies for content structure.
    They do not measure factual correctness or teaching accuracy.
    """
    rates = [
        safe_number(row, "introduction_indicators_per_1000_words"),
        safe_number(row, "explanation_indicators_per_1000_words"),
        safe_number(row, "flow_indicators_per_1000_words"),
        safe_number(row, "recap_indicators_per_1000_words"),
        safe_number(row, "conclusion_indicators_per_1000_words"),
    ]

    available = [r for r in rates if r is not None]

    if not available:
        return None

    # Provisional proxy: more detected structural cues contribute
    # to the score, capped to avoid unbounded values.
    component_scores = [
        clamp(rate * 20) for rate in available
    ]

    return average_available(component_scores)


# --------------------------------------------------
# 7. TEACHING STYLE SCORE
# --------------------------------------------------

def score_teaching_style(row):
    """
    Uses detected interaction and encouragement language.
    Individual attention, inclusiveness and behaviour need
    more detailed annotation or additional features.
    """
    interaction = safe_number(
        row, "interaction_indicators_per_1000_words"
    )
    encouragement = safe_number(
        row, "encouragement_indicators_per_1000_words"
    )

    components = []

    if interaction is not None:
        components.append(clamp(interaction * 20))

    if encouragement is not None:
        components.append(clamp(encouragement * 25))

    return average_available(components)


# --------------------------------------------------
# 8. ENGAGEMENT SCORE
# --------------------------------------------------

def score_engagement(row):
    questions = safe_number(
        row, "question_indicators_per_1000_words"
    )
    interaction = safe_number(
        row, "interaction_indicators_per_1000_words"
    )

    components = []

    if questions is not None:
        components.append(clamp(questions * 20))

    if interaction is not None:
        components.append(clamp(interaction * 20))

    return average_available(components)


# --------------------------------------------------
# 9. COMMUNICATION SCORE
# --------------------------------------------------

def score_communication(row):
    clarity = safe_number(
        row, "clarity_indicators_per_1000_words"
    )
    unique_ratio = safe_number(row, "unique_word_ratio")

    components = []

    if clarity is not None:
        components.append(clamp(clarity * 25))

    if unique_ratio is not None:
        # Vocabulary diversity is only a weak communication proxy.
        # It is not a direct measure of clarity or pronunciation.
        components.append(
            clamp(100 - abs(unique_ratio - 0.45) * 100)
        )

    return average_available(components)


# --------------------------------------------------
# 10. OVERALL WEIGHTED SCORE
# --------------------------------------------------

def calculate_overall(scores):
    """
    Calculate a weighted score from available dimensions only.
    The result is not final until the scoring rules are validated.
    """
    weighted_total = 0
    available_weight = 0

    for dimension, weight in WEIGHTS.items():
        score = scores.get(dimension)

        if score is not None:
            weighted_total += score * weight
            available_weight += weight

    if available_weight == 0:
        return None

    return round(weighted_total / available_weight, 2)


def score_recording(row):
    scores = {
        "audio_quality": score_audio(row),
        "visual_quality": score_visual(row),
        "content_quality": score_content(row),
        "teaching_style": score_teaching_style(row),
        "engagement": score_engagement(row),
        "communication": score_communication(row),
    }

    scores["overall_score"] = calculate_overall(scores)

    return scores


# --------------------------------------------------
# 11. RUN SCORING AND SAVE REPORT
# --------------------------------------------------

if __name__ == "__main__":
    try:
        data = load_and_match_data()

        results = []

        for _, row in data.iterrows():
            scores = score_recording(row)

            results.append({
                "recording": row.get("file_name"),
                "category": row.get("category"),
                "audio_video_match": row.get("match_status"),
                **scores,
            })

        results_df = pd.DataFrame(results)

        output_path = EVALUATION_DIR / "individual_scores.csv"
        results_df.to_csv(output_path, index=False)

        display_columns = [
            "recording",
            "category",
            "audio_quality",
            "visual_quality",
            "content_quality",
            "teaching_style",
            "engagement",
            "communication",
            "overall_score",
        ]

        print("\n--- INDIVIDUAL DIMENSION SCORES ---")
        print(results_df[display_columns].to_string(index=False))

        print("\nScores saved to:")
        print(output_path)

        print(
            "\nNOTE: These are provisional heuristic scores, "
            "not validated quality ratings."
        )

    except (FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"\nERROR: {error}")
