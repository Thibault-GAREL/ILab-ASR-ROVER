"""
Example: Using custom configuration
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MeetingTranscriptionPipeline
from src.utils.config_loader import setup_logging
import logging


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Custom configuration for each component
    diarizer_config = {
        "model_name": "pyannote/speaker-diarization-3.1",
        "device": "cuda",
        "min_speakers": 2,
        "max_speakers": 5,
        "hf_token": None  # Set your HF token or use environment variable
    }

    whisper_config = {
        "model_size": "large-v3",
        "device": "cuda",
        "compute_type": "float16",
    }

    canary_config = {
        "model_name": "nvidia/canary-1b",
        "device": "cuda",
        "decode_method": "greedy",
    }

    rover_config = {
        "voting_method": "confidence_weighted",
        "confidence_weights": {
            "whisper": 1.0,
            "canary": 1.3  # Give more weight to Canary (higher accuracy)
        },
        "min_confidence_threshold": 0.2,
        "word_error_tolerance": 0.15,
    }

    # Initialize pipeline with custom config
    pipeline = MeetingTranscriptionPipeline(
        diarizer_config=diarizer_config,
        whisper_config=whisper_config,
        canary_config=canary_config,
        rover_config=rover_config,
        use_canary=True
    )

    # Transcribe
    audio_file = "data/input/meeting.wav"
    result = pipeline.transcribe(
        audio_path=audio_file,
        language="fr",  # Force French
        min_speakers=2,
        max_speakers=4
    )

    # Save results
    pipeline.save_results(
        transcription=result,
        output_dir="data/output",
        base_name="custom_transcript",
        formats=["json", "txt", "srt"]
    )

    print(f"\nTranscription complete!")
    print(f"Duration: {result.duration:.2f}s")
    print(f"Speakers: {len(result.speakers)}")
    print(f"Segments: {len(result.segments)}")
    print(f"RTFx: {result.metadata['rtfx']:.2f}x")


if __name__ == "__main__":
    main()
