
from pathlib import Path
import re
import csv
import statistics
from collections import defaultdict


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "reference_videos"
OUTPUT_DIR = PROJECT_DIR / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. TRANSCRIPT FEATURE GROUPS
# --------------------------------------------------

KEYWORD_GROUPS = {
    "question_indicators": [
        "what", "why", "how", "which", "who", "when",
        "can you", "do you", "is it", "are you",
    ],

    "interaction_indicators": [
        "tell me", "answer me", "any questions",
        "do you understand", "everyone", "your turn",
        "can you explain", "what do you think",
    ],

    "explanation_indicators": [
        "because", "for example", "means", "that is",
        "in other words", "let us understand", "suppose",
    ],

    "encouragement_indicators": [
        "well done", "very good", "excellent", "great",
        "correct", "good job", "nice",
    ],

    "homework_indicators": [
        "homework", "assignment", "practice", "submit",
        "exercise", "revise",
    ],

    "recap_indicators": [
        "recap", "revision", "summarize", "summary",
        "today we learned", "last class", "in conclusion",
    ],

    "introduction_indicators": [
        "today we will learn",
        "today's topic",
        "in this class",
        "learning objectives",
        "previous class",
        "let us start",
    ],

    "flow_indicators": [
        "first",
        "next",
        "then",
        "moving on",
        "let us move to",
        "finally",
    ],

    "clarity_indicators": [
        "for example",
        "this means",
        "in simple words",
        "because",
        "in other words",
        "let me explain",
    ],

    "conclusion_indicators": [
        "to summarize",
        "in conclusion",
        "to conclude",
        "key points",
        "today we learned",
        "let us recap",
    ],
}


# --------------------------------------------------
# 3. READ AND CLEAN VTT TRANSCRIPTS
# --------------------------------------------------

def read_vtt(file_path):
    """Read a VTT file and remove timestamps and formatting."""

    content = file_path.read_text(
        encoding="utf-8-sig",
        errors="ignore",
    )

    # Remove timestamp lines
    content = re.sub(
        r"(?m)^.*\d{2}:\d{2}(?::\d{2})?\.\d+\s*-->.*$",
        " ",
        content,
    )

    # Remove HTML tags
    content = re.sub(r"<[^>]+>", " ", content)

    # Remove WEBVTT header and cue numbers
    content = re.sub(r"(?m)^\s*WEBVTT\s*$", " ", content)
    content = re.sub(r"(?m)^\s*\d+\s*$", " ", content)

    # Normalize whitespace
    content = re.sub(r"\s+", " ", content).strip()

    return content


# --------------------------------------------------
# 4. COUNT KEYWORD PHRASES
# --------------------------------------------------

def count_phrases(text, phrases):
    """Count phrase occurrences without case sensitivity."""

    text = text.lower()
    total = 0

    for phrase in phrases:
        pattern = (
            r"(?<!\w)"
            + re.escape(phrase.lower())
            + r"(?!\w)"
        )

        total += len(re.findall(pattern, text))

    return total


# --------------------------------------------------
# 5. ANALYZE ONE TRANSCRIPT
# --------------------------------------------------

def analyze_transcript(file_path, category):
    text = read_vtt(file_path)

    words = re.findall(r"\b[\w']+\b", text.lower())
    word_count = len(words)

    row = {
        "category": category,
        "transcript_file": file_path.name,
        "word_count": word_count,
        "unique_word_count": len(set(words)),
        "unique_word_ratio": round(
            len(set(words)) / word_count, 4
        ) if word_count else 0,
        "question_mark_count": text.count("?"),
    }

    # Count each feature group
    for feature_name, phrases in KEYWORD_GROUPS.items():
        count = count_phrases(text, phrases)

        row[feature_name] = count

        # Normalize by transcript length
        row[feature_name + "_per_1000_words"] = round(
            count / word_count * 1000, 2
        ) if word_count else 0

    return row


# --------------------------------------------------
# 6. SAVE CSV FILE
# --------------------------------------------------

def save_csv(file_path, rows):
    if not rows:
        print(f"No data to save: {file_path.name}")
        return

    with file_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------
# 7. MAIN PROGRAM
# --------------------------------------------------

def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Data folder not found: {DATA_DIR}")
        return

    records = []
    group_features = defaultdict(list)

    for category_dir in sorted(DATA_DIR.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name

        for file_path in sorted(category_dir.glob("*.vtt")):
            print(f"Analyzing: {category}/{file_path.name}")

            row = analyze_transcript(file_path, category)

            records.append(row)
            group_features[category].append(row)

    if not records:
        print(f"No VTT transcript files found in: {DATA_DIR}")
        return

    # Save individual transcript metrics
    features_file = OUTPUT_DIR / "transcript_features.csv"
    save_csv(features_file, records)

    # Build category-level averages
    summary_rows = []

    numeric_features = [
        "word_count",
        "unique_word_count",
        "unique_word_ratio",
        "question_mark_count",
    ]

    for feature_name in KEYWORD_GROUPS:
        numeric_features.append(feature_name)
        numeric_features.append(
            feature_name + "_per_1000_words"
        )

    for category, rows in sorted(group_features.items()):
        summary = {
            "category": category,
            "transcript_count": len(rows),
        }

        for feature in numeric_features:
            values = [row[feature] for row in rows]

            summary["mean_" + feature] = round(
                statistics.mean(values), 3
            ) if values else 0

        summary_rows.append(summary)

    # Save category averages
    summary_file = OUTPUT_DIR / "transcript_group_summary.csv"
    save_csv(summary_file, summary_rows)

    print("\n" + "=" * 50)
    print("TRANSCRIPT ANALYSIS COMPLETED")
    print("=" * 50)

    print(f"Transcripts analyzed: {len(records)}")
    print(f"Categories analyzed: {len(group_features)}")
    print(f"\nIndividual features:\n{features_file}")
    print(f"\nCategory summary:\n{summary_file}")

    print("\nTranscript count by category:")

    for category, rows in sorted(group_features.items()):
        print(f"{category}: {len(rows)}")


if __name__ == "__main__":
    main()
