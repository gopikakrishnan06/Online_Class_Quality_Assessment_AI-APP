
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

DATA_DIR = Path("data/reference_videos")
OUTPUT_DIR = Path("evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SILENCE_THRESHOLD = 0.01
CLIPPING_THRESHOLD = 0.99

audio_files = list(DATA_DIR.rglob("*_audio.wav"))
print("Audio files found:", len(audio_files))

results = []

for audio_path in audio_files:
    print(f"\nAnalyzing: {audio_path.name}")

    audio, sample_rate = sf.read(str(audio_path))

    if audio.ndim == 2:
        audio = audio.mean(axis=1)

    audio = audio.astype(np.float64)

    if audio.size == 0:
        print("Skipping empty audio file.")
        continue

    duration = len(audio) / sample_rate
    samples_per_second = sample_rate

    usable_length = (len(audio) // samples_per_second) * samples_per_second

    if usable_length == 0:
        print("Audio is shorter than one second.")
        continue

    segments = audio[:usable_length].reshape(-1, samples_per_second)
    rms_per_second = np.sqrt(np.mean(segments ** 2, axis=1))

    mean_rms = float(np.mean(rms_per_second))
    silence_percentage = float(
        np.mean(rms_per_second < SILENCE_THRESHOLD) * 100
    )

    peak_amplitude = float(np.max(np.abs(audio)))
    clipping_percentage = float(
        np.mean(np.abs(audio) >= CLIPPING_THRESHOLD) * 100
    )

    result = {
        "category": audio_path.parent.name,
        "file_name": audio_path.name,
        "duration_seconds": round(duration, 2),
        "sample_rate": sample_rate,
        "mean_rms_energy": round(mean_rms, 6),
        "silence_percentage": round(silence_percentage, 2),
        "peak_amplitude": round(peak_amplitude, 6),
        "clipping_percentage": round(clipping_percentage, 4),
    }

    results.append(result)

    print("Mean RMS energy:", result["mean_rms_energy"])
    print("Silence percentage:", result["silence_percentage"], "%")
    print("Peak amplitude:", result["peak_amplitude"])
    print("Clipping percentage:", result["clipping_percentage"], "%")

if results:
    df = pd.DataFrame(results)
    output_path = OUTPUT_DIR / "audio_features.csv"
    df.to_csv(output_path, index=False)

    print("\nAudio analysis completed.")
    print("Results saved to:", output_path)
    print(df.to_string(index=False))
else:
    print("No audio features were calculated.")
