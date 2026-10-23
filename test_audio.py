
from pathlib import Path
from moviepy import VideoFileClip

DATA_DIR = Path("data/reference_videos")

video_extensions = {".mp4", ".avi", ".mov", ".mkv"}

videos = [
    path
    for path in DATA_DIR.rglob("*")
    if path.is_file()
    and path.suffix.lower() in video_extensions
]

print("Videos found:", len(videos))

for video_path in videos:
    print(f"\nChecking: {video_path.name}")

    try:
        with VideoFileClip(str(video_path)) as clip:
            print("Duration:", round(clip.duration, 2), "seconds")
            print("Has audio:", clip.audio is not None)

            if clip.audio is not None:
                audio_path = video_path.with_name(
                    video_path.stem + "_audio.wav"
                )

                clip.audio.write_audiofile(
                    str(audio_path),
                    fps=16000,
                    logger=None
                )

                print("Audio saved:", audio_path)

    except Exception as error:
        print("Error:", error)
