#!/usr/bin/env python3
"""
Command-line interface for meeting transcription

Usage:
    python examples/cli_transcribe.py input.wav [--output OUTPUT_DIR] [--language LANG]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import MeetingTranscriptionPipeline
from src.utils.config_loader import load_config, setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transcribe meeting audio with speaker diarization"
    )

    parser.add_argument(
        "audio_file",
        type=str,
        help="Path to audio file"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/output",
        help="Output directory (default: data/output)"
    )

    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (en, fr, etc.) or auto-detect if not specified"
    )

    parser.add_argument(
        "--num-speakers",
        type=int,
        default=None,
        help="Number of speakers (auto-detect if not specified)"
    )

    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers"
    )

    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers"
    )

    parser.add_argument(
        "--whisper-only",
        action="store_true",
        help="Use Whisper only (disable Canary and ROVER)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: configs/config.yaml)"
    )

    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["json", "txt", "srt"],
        help="Output formats (json, txt, srt)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Check if audio file exists
    if not Path(args.audio_file).exists():
        print(f"Error: Audio file not found: {args.audio_file}")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    # Initialize pipeline
    print(f"Initializing transcription pipeline...")
    pipeline = MeetingTranscriptionPipeline(
        diarizer_config=config.get("diarization"),
        whisper_config=config.get("whisper"),
        canary_config=config.get("canary"),
        rover_config=config.get("rover"),
        use_canary=not args.whisper_only
    )

    # Transcribe
    print(f"\nTranscribing: {args.audio_file}")
    print(f"Language: {args.language or 'auto-detect'}")
    print(f"Speakers: {args.num_speakers or 'auto-detect'}")
    print(f"Mode: {'Whisper + Canary + ROVER' if not args.whisper_only else 'Whisper only'}")
    print()

    result = pipeline.transcribe(
        audio_path=args.audio_file,
        language=args.language,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )

    # Print summary
    print("\n" + "=" * 80)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 80)
    print(f"Duration: {result.duration:.2f}s")
    print(f"Language: {result.language}")
    print(f"Speakers: {', '.join(result.speakers)} ({len(result.speakers)} total)")
    print(f"Segments: {len(result.segments)}")
    print(f"Processing time: {result.metadata['processing_time']:.2f}s")
    print(f"Real-time factor: {result.metadata['rtfx']:.2f}x")
    print(f"Systems used: {', '.join(result.metadata['systems_used'])}")
    print("=" * 80 + "\n")

    # Save results
    base_name = Path(args.audio_file).stem + "_transcript"
    pipeline.save_results(
        transcription=result,
        output_dir=args.output,
        base_name=base_name,
        formats=args.formats
    )

    print(f"Results saved to: {args.output}/")
    print(f"Files created:")
    for fmt in args.formats:
        print(f"  - {base_name}.{fmt}")


if __name__ == "__main__":
    main()
