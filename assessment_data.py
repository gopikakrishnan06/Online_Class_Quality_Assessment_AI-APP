
from pathlib import Path
import pandas as pd


# ==================================================
# 1. PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
EVALUATION_DIR = PROJECT_DIR / "evaluation"

AUDIO_CSV = EVALUATION_DIR / "audio_features.csv"
VIDEO_CSV = EVALUATION_DIR / "visual_video_summary.csv"
TRANSCRIPT_CSV = EVALUATION_DIR / "transcript_features.csv"


# ==================================================
# 2. LOAD EXISTING ANALYSIS RESULTS
# ==================================================

def load_analysis_data():
    """Load audio, video and transcript analysis CSV files."""

    required_files = [
        AUDIO_CSV,
        VIDEO_CSV,
        TRANSCRIPT_CSV,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required CSV file not found: {file_path}"
            )

    audio_df = pd.read_csv(AUDIO_CSV)
    video_df = pd.read_csv(VIDEO_CSV)
    transcript_df = pd.read_csv(TRANSCRIPT_CSV)

    print("Analysis data loaded successfully.")

    return audio_df, video_df, transcript_df


# ==================================================
# 3. NORMALIZE FILENAMES FOR MATCHING
# ==================================================

def normalize_audio_filename(filename):
    """Remove extension and the trailing _audio suffix."""

    name = Path(str(filename)).stem.strip()

    if name.casefold().endswith("_audio"):
        name = name[:-6]

    return name.strip().casefold()


def normalize_video_filename(filename):
    """Remove extension and normalize the video filename."""

    return Path(str(filename)).stem.strip().casefold()


def normalize_transcript_filename(filename):
    """
    Normalize transcript filenames for candidate matching.

    Example:
    B435 intermediate Part -transcript-1.1.vtt
    becomes:
    b435 intermediate part
    """

    name = Path(str(filename)).stem.strip().casefold()

    # Remove common transcript suffixes.
    for suffix in (
        "-transcript-1.1 (1)",
        "-transcript-1.1",
        "-transcript",
    ):
        if name.endswith(suffix):
            name = name[:-len(suffix)].strip()
            break

    return name


# ==================================================
# 4. MATCH AUDIO WITH VIDEO
# ==================================================

def match_audio_video(audio_df, video_df):
    """Match audio and video using normalized filenames."""

    matched_records = []

    for _, audio in audio_df.iterrows():

        audio_key = normalize_audio_filename(
            audio["file_name"]
        )

        video_matches = video_df[
            video_df["video_name"].apply(
                normalize_video_filename
            ) == audio_key
        ]

        if len(video_matches) == 1:

            video = video_matches.iloc[0]

            matched_records.append({
                "audio_file": audio["file_name"],
                "video_file": video["video_name"],
                "category": audio["category"],
                "match_status": "Matched",

                # Audio features
                "duration_seconds": audio["duration_seconds"],
                "mean_rms_energy": audio["mean_rms_energy"],
                "silence_percentage": audio["silence_percentage"],
                "peak_amplitude": audio["peak_amplitude"],
                "clipping_percentage": audio["clipping_percentage"],

                # Video features
                "mean_brightness": video["mean_brightness"],
                "brightness_variation": video["brightness_variation"],
                "mean_contrast": video["mean_contrast"],
                "mean_sharpness": video["mean_sharpness"],
                "dark_frame_percentage": video["dark_frame_percentage"],
                "overexposed_frame_percentage":
                    video["potential_overexposed_frame_percentage"],
                "freeze_suspect_count": video["freeze_suspect_count"],
                "video_analysis_status": video["analysis_status"],
            })

        elif len(video_matches) == 0:
            print(
                f"WARNING: No video match for "
                f"{audio['file_name']}"
            )

        else:
            print(
                f"WARNING: Multiple video matches for "
                f"{audio['file_name']}"
            )

    return pd.DataFrame(matched_records)


# ==================================================
# 5. FIND TRANSCRIPT CANDIDATES
# ==================================================

def find_transcript_candidates(
    matched_df,
    transcript_df,
):
    """
    Find transcript candidates using the recording name
    and category.

    Important:
    Candidate matching does not prove that the transcript
    belongs to the exact dated recording.
    """

    results = []

    for _, recording in matched_df.iterrows():

        recording_key = normalize_audio_filename(
            recording["audio_file"]
        )

        # Remove the date portion to get the shared
        # recording identity, e.g. B435 intermediate Part.
        parts = recording_key.rsplit(" - ", 1)

        if len(parts) == 2:
            recording_identity = parts[0].strip()
        else:
            recording_identity = recording_key

        category = str(recording["category"]).strip().casefold()

        candidates = transcript_df[
            transcript_df["category"].astype(str).str.strip().str.casefold()
            == category
        ].copy()

        candidates["_normalized_name"] = (
            candidates["transcript_file"]
            .apply(normalize_transcript_filename)
        )

        # Match the shared recording identity, not just category.
        candidates = candidates[
            candidates["_normalized_name"].apply(
                lambda name: (
                    name == recording_identity
                    or name.startswith(recording_identity + " ")
                    or recording_identity.startswith(name + " ")
                )
            )
        ]

        if len(candidates) == 1:
            transcript = candidates.iloc[0]

            results.append({
                "audio_file": recording["audio_file"],
                "transcript_file": transcript["transcript_file"],
                "transcript_category": transcript["category"],
                "transcript_match_status": "Candidate found - verify identity",
            })

        elif len(candidates) == 0:
            results.append({
                "audio_file": recording["audio_file"],
                "transcript_file": None,
                "transcript_category": None,
                "transcript_match_status": "No candidate found",
            })

        else:
            results.append({
                "audio_file": recording["audio_file"],
                "transcript_file": "; ".join(
                    candidates["transcript_file"].astype(str)
                ),
                "transcript_category": category,
                "transcript_match_status": "Multiple candidates - manual verification",
            })

    return pd.DataFrame(results)


# ==================================================
# 6. RUN AND VALIDATE
# ==================================================

if __name__ == "__main__":

    try:
        audio_df, video_df, transcript_df = load_analysis_data()

        print("\n--- DATASET COUNTS ---")
        print("Audio records:", len(audio_df))
        print("Video records:", len(video_df))
        print("Transcript records:", len(transcript_df))

        # Match audio and video.
        matched_df = match_audio_video(
            audio_df,
            video_df,
        )

        print("\n--- AUDIO-VIDEO MATCHING ---")

        if matched_df.empty:
            print("No audio-video matches found.")
        else:
            print(
                matched_df[
                    [
                        "audio_file",
                        "video_file",
                        "category",
                        "match_status",
                    ]
                ].to_string(index=False)
            )

        print(
            "\nTotal matched recordings:",
            len(matched_df),
        )

        # Find transcript candidates.
        transcript_matches = find_transcript_candidates(
            matched_df,
            transcript_df,
        )

        print("\n--- TRANSCRIPT CANDIDATES ---")

        if transcript_matches.empty:
            print("No transcript candidates found.")
        else:
            print(transcript_matches.to_string(index=False))

        # Save combined audio-video features.
        output_path = EVALUATION_DIR / "combined_features.csv"
        matched_df.to_csv(output_path, index=False)

        print("\nCombined features saved to:")
        print(output_path)

    except (FileNotFoundError, KeyError, pd.errors.ParserError) as error:
        print(f"\nERROR: {error}")
