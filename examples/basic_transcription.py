"""
Basic example: Transcribe a meeting with speaker diarization
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MeetingTranscriptionPipeline
from src.utils.config_loader import load_config, setup_logging


def main():
    # Load configuration
    config = load_config()
    setup_logging(config)

    # Initialize pipeline
    pipeline = MeetingTranscriptionPipeline(
        diarizer_config=config.get("diarization"),
        whisper_config=config.get("whisper"),
        canary_config=config.get("canary"),
        rover_config=config.get("rover"),
        use_canary=True  # Set to False if Canary not available
    )

    # Input audio file
    audio_file = "data/input/meeting.wav"

    # Transcribe
    print(f"Transcribing: {audio_file}")
    result = pipeline.transcribe(
        audio_path=audio_file,
        language=None,  # Auto-detect (or specify "en", "fr")
        num_speakers=None,  # Auto-detect number of speakers
    )

    # Print results
    print("\n" + "=" * 80)
    print(f"Meeting Transcription Results")
    print("=" * 80)
    print(f"Duration: {result.duration:.2f}s")
    print(f"Language: {result.language}")
    print(f"Speakers: {', '.join(result.speakers)}")
    print(f"Number of segments: {len(result.segments)}")
    print(f"Processing time: {result.metadata['processing_time']:.2f}s")
    print(f"RTFx: {result.metadata['rtfx']:.2f}x")
    print("\n" + "=" * 80)
    print("Transcript:")
    print("=" * 80 + "\n")

    for segment in result.segments:
        print(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.speaker}:")
        print(f"  {segment.text}")
        print(f"  (confidence: {segment.confidence:.3f})")
        print()

    # Save results
    pipeline.save_results(
        transcription=result,
        output_dir="data/output",
        base_name="meeting_transcript",
        formats=["json", "txt", "srt"]
    )

    print("\nResults saved to data/output/")


if __name__ == "__main__":
    main()
