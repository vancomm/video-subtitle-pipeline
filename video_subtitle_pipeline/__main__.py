import io
import math
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment
import ffmpeg


def extract_audio(video_path: Path, audio_path: Path) -> None:
    video_stream = ffmpeg.input(str(video_path))
    audio_stream = ffmpeg.output(video_stream, str(audio_path))
    ffmpeg.run(audio_stream, overwrite_output=True)


def transcribe(audio_path: Path, model: WhisperModel) -> tuple[str, list[Segment]]:
    segments, info = model.transcribe(str(audio_path))
    segments = list(segments)
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    return info.language, segments


def format_time(seconds: float) -> str:
    hours, seconds = math.floor(seconds / 3600), seconds % 3600
    minutes, seconds = math.floor(seconds / 60), seconds % 60
    milliseconds = round((seconds - math.floor(seconds)) * 1000)
    seconds = math.floor(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:01d},{milliseconds:03d}"


def generate_subtitle_text(segments: list[Segment]) -> str:
    buffer = io.StringIO()
    for index, segment in enumerate(segments):
        segment_start = format_time(segment.start)
        segment_end = format_time(segment.end)
        subtitle = f"{index+1}\n{segment_start} --> {segment_end}\n{segment.text}\n\n"
        buffer.write(subtitle)
    return buffer.getvalue()


def add_subtitle_to_video(
    subtitle_path: Path,
    in_video_path: Path,
    out_video_path: Path,
    *,
    soft: bool = True,
    language: str,
) -> None:
    in_video_stream = ffmpeg.input(str(in_video_path))

    if soft:
        subtitle_stream = ffmpeg.input(str(subtitle_path))
        out_video_stream = ffmpeg.output(
            in_video_stream,
            subtitle_stream,
            str(out_video_path),
            **{"c": "copy", "c:s": "mov_text"},
            **{
                "metadata:s:s:0": f"language={language}",
                "metadata:s:s:0": f"title={subtitle_path.stem}",
            },
        )
    else:
        out_video_stream = ffmpeg.output(
            in_video_stream,
            str(out_video_path),
            vf=f"subtitles={subtitle_path}",
        )

    ffmpeg.run(out_video_stream, overwrite_output=True)


def main() -> None:
    video_path = Path("./data/dQw4w9WgXcQ-960.mp4")
    audio_path = video_path.parent / f"audio-{video_path.stem}.wav"

    if not audio_path.exists():
        extract_audio(video_path, audio_path)

    model_size = "small"
    model = WhisperModel(model_size, device="cpu")
    language, segments = transcribe(audio_path, model)

    subtitle_path = video_path.parent / f"sub-{video_path.stem}.{language}.srt"
    subtitle_text = generate_subtitle_text(segments)
    subtitle_path.write_text(subtitle_text)

    add_subtitle_to_video(
        subtitle_path,
        video_path,
        video_path.parent / f"subbed-{video_path.name}",
        language=language,
    )
    add_subtitle_to_video(
        subtitle_path,
        video_path,
        video_path.parent / f"hardsubbed-{video_path.name}",
        soft=False,
        language=language,
    )


if __name__ == "__main__":
    main()
