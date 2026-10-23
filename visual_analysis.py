
from pathlib import Path
import csv
import cv2
import numpy as np

# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FRAME_CSV = OUTPUT_DIR / "visual_frame_features.csv"
SUMMARY_CSV = OUTPUT_DIR / "visual_video_summary.csv"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

# Analyze one frame every 5 seconds
SAMPLE_INTERVAL_SECONDS = 5


# --------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------

def get_lighting_status(brightness):
    if brightness < 50:
        return "Too Dark"
    elif brightness > 210:
        return "Potentially Overexposed"
    return "Brightness Within Initial Range"


def calculate_frame_features(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resolution
    height, width = gray.shape

    # Average brightness: 0 to 255
    brightness = float(np.mean(gray))

    # Contrast indicator
    contrast = float(np.std(gray))

    # Sharpness indicator
    sharpness = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    return {
        "width": width,
        "height": height,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "lighting_status": get_lighting_status(brightness),
    }


# --------------------------------------------------
# 3. FIND VIDEOS IN ALL FOUR GROUPS
# --------------------------------------------------

video_files = sorted(
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in VIDEO_EXTENSIONS
)

print(f"Videos found: {len(video_files)}")

if not video_files:
    raise FileNotFoundError(
        f"No supported videos found inside: {DATA_DIR}"
    )


# --------------------------------------------------
# 4. CSV COLUMN DEFINITIONS
# --------------------------------------------------

frame_columns = [
    "video_name",
    "group_label",
    "timestamp_seconds",
    "width",
    "height",
    "brightness",
    "contrast",
    "sharpness",
    "lighting_status",
]

summary_columns = [
    "video_name",
    "group_label",
    "duration_seconds",
    "fps",
    "reported_frame_count",
    "sampled_frames",
    "width",
    "height",
    "mean_brightness",
    "brightness_variation",
    "mean_contrast",
    "mean_sharpness",
    "dark_frame_percentage",
    "potential_overexposed_frame_percentage",
    "estimated_frame_gap_count",
    "freeze_suspect_count",
    "analysis_status",
]


# --------------------------------------------------
# 5. ANALYZE EACH VIDEO
# --------------------------------------------------

with open(FRAME_CSV, "w", newline="", encoding="utf-8-sig") as frame_file, \
     open(SUMMARY_CSV, "w", newline="", encoding="utf-8-sig") as summary_file:

    frame_writer = csv.DictWriter(
        frame_file, fieldnames=frame_columns
    )
    summary_writer = csv.DictWriter(
        summary_file, fieldnames=summary_columns
    )

    frame_writer.writeheader()
    summary_writer.writeheader()

    for video_path in video_files:
        print(f"\nAnalyzing: {video_path.name}")

        # Assumes each video's parent folder is its group label
        group_label = video_path.parent.name

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            print("Could not open video.")
            summary_writer.writerow({
                "video_name": video_path.name,
                "group_label": group_label,
                "analysis_status": "Could not open video",
            })
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        reported_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        duration = (
            reported_frames / fps
            if fps and fps > 0
            else 0
        )

        interval_frames = (
            max(1, round(fps * SAMPLE_INTERVAL_SECONDS))
            if fps and fps > 0
            else 1
        )

        brightness_values = []
        contrast_values = []
        sharpness_values = []
        timestamps = []
        dark_count = 0
        bright_count = 0
        freeze_suspect_count = 0
        sampled_count = 0
        width = 0
        height = 0

        previous_gray = None
        previous_timestamp = None
        estimated_frame_gap_count = 0
        frame_index = 0

        while True:
            success, frame = cap.read()

            if not success:
                break

            # Sample frames at regular intervals
            if frame_index % interval_frames == 0:
                timestamp = (
                    frame_index / fps
                    if fps and fps > 0
                    else float(frame_index)
                )

                features = calculate_frame_features(frame)
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )

                width = features["width"]
                height = features["height"]

                brightness = features["brightness"]
                contrast = features["contrast"]
                sharpness = features["sharpness"]

                brightness_values.append(brightness)
                contrast_values.append(contrast)
                sharpness_values.append(sharpness)
                timestamps.append(timestamp)
                sampled_count += 1

                if brightness < 50:
                    dark_count += 1

                if brightness > 210:
                    bright_count += 1

                # Check unusually large timestamp gaps
                if previous_timestamp is not None:
                    if (
                        timestamp - previous_timestamp
                        > SAMPLE_INTERVAL_SECONDS * 1.5
                    ):
                        estimated_frame_gap_count += 1

                # Similarity check: flag near-identical
                # consecutive sampled frames for review
                if previous_gray is not None:
                    difference = cv2.absdiff(
                        previous_gray, gray
                    )
                    mean_difference = float(
                        np.mean(difference)
                    )

                    if mean_difference < 0.5:
                        freeze_suspect_count += 1

                previous_gray = gray
                previous_timestamp = timestamp

                frame_writer.writerow({
                    "video_name": video_path.name,
                    "group_label": group_label,
                    "timestamp_seconds": round(timestamp, 2),
                    **features,
                })

            frame_index += 1

        cap.release()

        # Calculate per-video statistics
        if sampled_count > 0:
            mean_brightness = float(
                np.mean(brightness_values)
            )
            brightness_variation = float(
                np.std(brightness_values)
            )
            mean_contrast = float(
                np.mean(contrast_values)
            )
            mean_sharpness = float(
                np.mean(sharpness_values)
            )

            dark_percentage = (
                dark_count / sampled_count
            ) * 100

            overexposed_percentage = (
                bright_count / sampled_count
            ) * 100

            status = "Success"

        else:
            mean_brightness = 0
            brightness_variation = 0
            mean_contrast = 0
            mean_sharpness = 0
            dark_percentage = 0
            overexposed_percentage = 0
            status = "No frames sampled"

        summary_writer.writerow({
            "video_name": video_path.name,
            "group_label": group_label,
            "duration_seconds": round(duration, 2),
            "fps": round(fps, 2) if fps > 0 else 0,
            "reported_frame_count": reported_frames,
            "sampled_frames": sampled_count,
            "width": width,
            "height": height,
            "mean_brightness": round(mean_brightness, 2),
            "brightness_variation": round(
                brightness_variation, 2
            ),
            "mean_contrast": round(mean_contrast, 2),
            "mean_sharpness": round(mean_sharpness, 2),
            "dark_frame_percentage": round(
                dark_percentage, 2
            ),
            "potential_overexposed_frame_percentage": round(
                overexposed_percentage, 2
            ),
            "estimated_frame_gap_count":
                estimated_frame_gap_count,
            "freeze_suspect_count": freeze_suspect_count,
            "analysis_status": status,
        })

        print(
            f"Sampled frames: {sampled_count} | "
            f"Brightness: {mean_brightness:.2f} | "
            f"Sharpness: {mean_sharpness:.2f}"
        )


print("\nVisual analysis completed.")
print(f"Frame results: {FRAME_CSV}")
print(f"Video summaries: {SUMMARY_CSV}")
