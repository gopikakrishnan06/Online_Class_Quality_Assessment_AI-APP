
from pathlib import Path
from faster_whisper import WhisperModel

# 1. Define input and output folders
DATA_DIR = Path("data/reference_videos")
OUTPUT_DIR = Path("evaluation/transcripts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load the pre-trained speech recognition model
print("Loading Whisper model...")
model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

# 3. Find audio files
audio_files = list(DATA_DIR.rglob("*_audio.wav"))

print("Audio files found:", len(audio_files))

if not audio_files:
    raise FileNotFoundError(
        f"No audio files found in {DATA_DIR}"
    )

# 4. Transcribe each audio file
for audio_path in audio_files:
    print(f"\nTranscribing: {audio_path.name}")

    segments, info = model.transcribe(
        str(audio_path),
        beam_size=5
    )

    transcript_lines = []

    for segment in segments:
        line = (
            f"[{segment.start:.2f}s - {segment.end:.2f}s] "
            f"{segment.text.strip()}"
        )
        transcript_lines.append(line)
        print(line)

    # 5. Save transcript as a text file
    output_path = OUTPUT_DIR / f"{audio_path.stem}.txt"

    output_path.write_text(
        "\n".join(transcript_lines),
        encoding="utf-8"
    )

    print("Transcript saved to:", output_path)
    print("Detected language:", info.language)

print("\nAll transcriptions completed.")

