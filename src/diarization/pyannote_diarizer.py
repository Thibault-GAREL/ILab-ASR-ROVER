"""
Pyannote-based speaker diarization module
Supports pyannote.audio 3.1 and Community-1 models
"""

import logging
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class SpeakerSegment:
    """Represents a speech segment with speaker information"""
    start: float
    end: float
    speaker: str
    confidence: float = 1.0

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "confidence": self.confidence,
            "duration": self.duration()
        }


class PyannoteDiarizer:
    """
    Speaker diarization using Pyannote.audio

    Supports:
    - pyannote/speaker-diarization-3.1
    - pyannote/speaker-diarization-community-1 (recommended)
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        device: str = "cuda",
        hf_token: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """
        Initialize the diarization pipeline

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
            hf_token: HuggingFace API token (required for pyannote models)
            min_speakers: Minimum number of speakers (None for auto)
            max_speakers: Maximum number of speakers (None for auto)
        """
        self.model_name = model_name
        self.device = device if torch.cuda.is_available() else "cpu"
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        # Get HF token from parameter or environment
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        if not self.hf_token:
            logger.warning(
                "No HuggingFace token provided. "
                "Set HF_TOKEN environment variable or pass hf_token parameter."
            )

        logger.info(f"Initializing Pyannote diarization: {model_name}")
        self._load_pipeline()

    def _load_pipeline(self):
        """Load the diarization pipeline"""
        try:
            self.pipeline = Pipeline.from_pretrained(
                self.model_name,
                token=self.hf_token
            )

            if self.device == "cuda" and torch.cuda.is_available():
                self.pipeline.to(torch.device("cuda"))
                logger.info(f"Pipeline loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info("Pipeline loaded on CPU")

        except Exception as e:
            logger.error(f"Failed to load diarization pipeline: {e}")
            raise

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> List[SpeakerSegment]:
        """
        Perform speaker diarization on an audio file

        Args:
            audio_path: Path to audio file
            num_speakers: Exact number of speakers (overrides min/max)
            min_speakers: Minimum speakers (overrides init value)
            max_speakers: Maximum speakers (overrides init value)

        Returns:
            List of SpeakerSegment objects
        """
        logger.info(f"Diarizing: {audio_path}")

        # Determine speaker parameters
        min_spk = min_speakers or self.min_speakers
        max_spk = max_speakers or self.max_speakers

        try:
            # Run diarization
            if num_speakers:
                diarization = self.pipeline(
                    audio_path,
                    num_speakers=num_speakers
                )
            else:
                diarization = self.pipeline(
                    audio_path,
                    min_speakers=min_spk,
                    max_speakers=max_spk
                )

            # Convert to SpeakerSegment objects
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segment = SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                    confidence=1.0  # Pyannote doesn't provide segment-level confidence
                )
                segments.append(segment)

            # Log statistics
            unique_speakers = len(set(seg.speaker for seg in segments))
            total_duration = sum(seg.duration() for seg in segments)

            logger.info(
                f"Diarization complete: {len(segments)} segments, "
                f"{unique_speakers} speakers, {total_duration:.2f}s total speech"
            )

            return segments

        except Exception as e:
            logger.error(f"Diarization failed: {e}")
            raise

    def get_speaker_segments_dict(
        self,
        segments: List[SpeakerSegment]
    ) -> Dict[str, List[Tuple[float, float]]]:
        """
        Group segments by speaker

        Args:
            segments: List of SpeakerSegment objects

        Returns:
            Dictionary mapping speaker ID to list of (start, end) tuples
        """
        speaker_dict = {}
        for seg in segments:
            if seg.speaker not in speaker_dict:
                speaker_dict[seg.speaker] = []
            speaker_dict[seg.speaker].append((seg.start, seg.end))

        return speaker_dict

    def get_speaker_at_time(
        self,
        segments: List[SpeakerSegment],
        timestamp: float
    ) -> Optional[str]:
        """
        Find which speaker is speaking at a given timestamp

        Args:
            segments: List of SpeakerSegment objects
            timestamp: Time in seconds

        Returns:
            Speaker ID or None if no speaker at that time
        """
        for seg in segments:
            if seg.start <= timestamp <= seg.end:
                return seg.speaker
        return None

    def merge_close_segments(
        self,
        segments: List[SpeakerSegment],
        gap_threshold: float = 0.5
    ) -> List[SpeakerSegment]:
        """
        Merge segments from the same speaker that are close together

        Args:
            segments: List of SpeakerSegment objects
            gap_threshold: Maximum gap in seconds to merge

        Returns:
            List of merged SpeakerSegment objects
        """
        if not segments:
            return []

        # Sort by start time
        sorted_segments = sorted(segments, key=lambda x: x.start)
        merged = [sorted_segments[0]]

        for current in sorted_segments[1:]:
            previous = merged[-1]

            # Check if same speaker and close enough
            if (current.speaker == previous.speaker and
                current.start - previous.end <= gap_threshold):
                # Merge segments
                previous.end = current.end
            else:
                merged.append(current)

        logger.info(
            f"Merged {len(segments)} segments into {len(merged)} "
            f"(gap threshold: {gap_threshold}s)"
        )

        return merged

    def export_rttm(
        self,
        segments: List[SpeakerSegment],
        output_path: str,
        file_id: str = "audio"
    ):
        """
        Export diarization to RTTM format

        Args:
            segments: List of SpeakerSegment objects
            output_path: Path to save RTTM file
            file_id: File identifier for RTTM format
        """
        with open(output_path, 'w') as f:
            for seg in segments:
                # RTTM format: SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker> <NA> <NA>
                f.write(
                    f"SPEAKER {file_id} 1 {seg.start:.3f} {seg.duration():.3f} "
                    f"<NA> <NA> {seg.speaker} <NA> <NA>\n"
                )

        logger.info(f"RTTM exported to: {output_path}")
