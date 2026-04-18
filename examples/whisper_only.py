"""
Example: Using Whisper only (without Canary)
Useful if you don't have access to NVIDIA NeMo or Canary models
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MeetingTranscriptionPipeline
import logging


def main():
    logging.basicConfig(level=logging.INFO)

    # Initialize pipeline with Whisper only
    pipeline = MeetingTranscriptionPipeline(
        diarizer_config={
            "model_name": "pyannote/speaker-diarization-3.1",
            "device": "cuda",
        },
        whisper_config={
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
        },
        use_canary=False  # Disable Canary
    )

    # Transcribe
    audio_file = "data/input/meeting.wav"
    result = pipeline.transcribe(
        audio_path=audio_file,
        language=None,  # Auto-detect
    )

    # Display results
    print("\n" + "=" * 80)
    print("Whisper-Only Transcription")
    print("=" * 80)
    print(f"Language: {result.language}")
    print(f"Speakers: {', '.join(result.speakers)}")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Processing time: {result.metadata['processing_time']:.2f}s")
    print("=" * 80 + "\n")

    for i, segment in enumerate(result.segments[:5], 1):  # Show first 5 segments
        print(f"{i}. [{segment.start:.1f}s] {segment.speaker}: {segment.text[:100]}...")

    # Save
    pipeline.save_results(
        transcription=result,
        output_dir="data/output",
        base_name="whisper_only_transcript",
        formats=["json", "txt"]
    )


if __name__ == "__main__":
    main()
