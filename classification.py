
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent
SCORES_FILE = PROJECT_DIR / "evaluation" / "individual_scores.csv"

DIMENSIONS = [
    "audio_quality",
    "visual_quality",
    "content_quality",
]


def calculate_aggregate(row):
    scores = row[DIMENSIONS]

    # Do not classify if any required score is missing
    if scores.isna().any():
        return None

    if not scores.between(0, 100).all():
        return None

    return scores.mean()


def classify_score(score):
    if pd.isna(score):
        return "Not Available"
    elif score >= 85:
        return "Excellent"
    elif score >= 70:
        return "Good"
    elif score >= 50:
        return "Average"
    else:
        return "Poor"


def main():
    if not SCORES_FILE.exists():
        raise FileNotFoundError(
            f"Scores file not found: {SCORES_FILE}"
        )

    df = pd.read_csv(SCORES_FILE)

    missing = [c for c in DIMENSIONS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df["aggregate_score"] = df.apply(
        calculate_aggregate, axis=1
    )

    df["aggregate_category"] = df["aggregate_score"].apply(
        classify_score
    )

    columns = [
        "recording",
        *DIMENSIONS,
        "aggregate_score",
        "aggregate_category",
    ]

    print("\n--- Audio-Visual-Content Classification ---")
    print(df[columns].to_string(index=False))

    print(
        "\nNote: Categories use provisional prototype thresholds, "
        "not validated teaching-quality standards."
    )


if __name__ == "__main__":
    main()
