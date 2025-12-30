"""
Main pipeline for multilingual meeting transcription
Combines diarization, multiple ASR systems, and ROVER fusion
"""

import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import time

from .diarization.pyannote_diarizer import PyannoteDiarizer, SpeakerSegment
from .asr.whisper_asr import WhisperASR
from .asr.canary_asr import CanaryASR
from .asr.base_asr import ASRResult
from .rover.rover_fusion import ROVERFusion, FusionResult

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionSegment:
    """A transcription segment with speaker and timing"""
    start: float
    end: float
    speaker: str
    text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker,
            "text": self.text,
            "confidence": self.confidence,
            "duration": self.end - self.start,
            "metadata": self.metadata
        }


@dataclass
class MeetingTranscription:
    """Complete meeting transcription with diarization"""
    segments: List[TranscriptionSegment]
    full_text: str
    speakers: List[str]
    duration: float
    language: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "full_text": self.full_text,
            "speakers": self.speakers,
            "duration": self.duration,
            "language": self.language,
            "num_segments": len(self.segments),
            "metadata": self.metadata
        }

    def export_txt(self, output_path: str):
        """Export as plain text with speaker labels"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"Meeting Transcription\n")
            f.write(f"Duration: {self.duration:.2f}s\n")
            f.write(f"Language: {self.language}\n")
            f.write(f"Speakers: {', '.join(self.speakers)}\n")
            f.write("=" * 80 + "\n\n")

            for segment in self.segments:
                timestamp = f"[{segment.start:.2f}s - {segment.end:.2f}s]"
                f.write(f"{segment.speaker} {timestamp}:\n{segment.text}\n\n")

    def export_srt(self, output_path: str):
        """Export as SRT subtitle format"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(self.segments, 1):
                start_time = self._format_srt_time(segment.start)
                end_time = self._format_srt_time(segment.end)

                f.write(f"{i}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"[{segment.speaker}] {segment.text}\n\n")

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


class MeetingTranscriptionPipeline:
    """
    Complete pipeline for meeting transcription

    Pipeline stages:
    1. Speaker diarization (Pyannote)
    2. ASR System 1 (Whisper) - robustness
    3. ASR System 2 (Canary) - accuracy
    4. ROVER fusion
    5. Combine with diarization
    """

    def __init__(
        self,
        diarizer_config: Optional[Dict] = None,
        whisper_config: Optional[Dict] = None,
        canary_config: Optional[Dict] = None,
        rover_config: Optional[Dict] = None,
        use_canary: bool = True,
    ):
        """
        Initialize the pipeline

        Args:
            diarizer_config: Configuration for Pyannote diarizer
            whisper_config: Configuration for Whisper ASR
            canary_config: Configuration for Canary ASR
            rover_config: Configuration for ROVER fusion
            use_canary: Whether to use Canary (set False if not available)
        """
        self.use_canary = use_canary

        # Initialize diarization
        diarizer_config = diarizer_config or {}
        logger.info("Initializing speaker diarization...")
        self.diarizer = PyannoteDiarizer(**diarizer_config)

        # Initialize Whisper ASR
        whisper_config = whisper_config or {}
        logger.info("Initializing Whisper ASR...")
        self.whisper = WhisperASR(**whisper_config)

        # Initialize Canary ASR (optional)
        if self.use_canary:
            canary_config = canary_config or {}
            logger.info("Initializing Canary ASR...")
            try:
                self.canary = CanaryASR(**canary_config)
            except Exception as e:
                logger.warning(f"Failed to load Canary: {e}. Continuing with Whisper only.")
                self.use_canary = False
                self.canary = None
        else:
            self.canary = None

        # Initialize ROVER fusion
        if self.use_canary:
            rover_config = rover_config or {}
            logger.info("Initializing ROVER fusion...")
            self.rover = ROVERFusion(**rover_config)
        else:
            self.rover = None

        logger.info("Pipeline initialization complete")

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> MeetingTranscription:
        """
        Transcribe a meeting audio file

        Args:
            audio_path: Path to audio file
            language: Language code ("en", "fr", None for auto)
            num_speakers: Exact number of speakers (None for auto)
            min_speakers: Minimum speakers (None for auto)
            max_speakers: Maximum speakers (None for auto)

        Returns:
            MeetingTranscription object
        """
        start_time = time.time()
        logger.info(f"Starting meeting transcription: {audio_path}")

        # Stage 1: Speaker diarization
        logger.info("Stage 1/4: Speaker diarization")
        diarization_start = time.time()
        speaker_segments = self.diarizer.diarize(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        diarization_time = time.time() - diarization_start
        logger.info(f"Diarization complete in {diarization_time:.2f}s")

        # Stage 2: Whisper transcription
        logger.info("Stage 2/4: Whisper ASR")
        whisper_start = time.time()
        whisper_result = self.whisper.transcribe(audio_path, language=language)
        whisper_time = time.time() - whisper_start
        logger.info(f"Whisper transcription complete in {whisper_time:.2f}s")

        # Stage 3: Canary transcription (if available)
        canary_result = None
        canary_time = 0
        if self.use_canary and self.canary:
            logger.info("Stage 3/4: Canary ASR")
            canary_start = time.time()
            try:
                canary_result = self.canary.transcribe(audio_path, language=language)
                canary_time = time.time() - canary_start
                logger.info(f"Canary transcription complete in {canary_time:.2f}s")
            except Exception as e:
                logger.warning(f"Canary transcription failed: {e}. Using Whisper only.")

        # Stage 4: ROVER fusion
        fusion_time = 0
        if canary_result and self.rover:
            logger.info("Stage 4/4: ROVER fusion")
            fusion_start = time.time()
            asr_results = {
                "whisper": whisper_result,
                "canary": canary_result
            }
            fusion_result = self.rover.fuse(asr_results)
            fusion_time = time.time() - fusion_start
            logger.info(f"ROVER fusion complete in {fusion_time:.2f}s")

            # Use fused result
            final_text = fusion_result.text
            final_words = fusion_result.words
            final_confidence = fusion_result.confidence
        else:
            # Use Whisper only
            logger.info("Using Whisper result only (no ROVER fusion)")
            final_text = whisper_result.text
            final_words = whisper_result.words
            final_confidence = whisper_result.confidence

        # Combine transcription with diarization
        logger.info("Combining transcription with speaker diarization")
        transcription_segments = self._combine_asr_and_diarization(
            speaker_segments, final_words, final_text
        )

        # Get unique speakers
        unique_speakers = sorted(set(seg.speaker for seg in transcription_segments))

        # Calculate total duration
        duration = max(seg.end for seg in speaker_segments) if speaker_segments else 0

        # Detect language
        detected_language = language or whisper_result.language

        # Build metadata
        total_time = time.time() - start_time
        metadata = {
            "audio_path": audio_path,
            "processing_time": total_time,
            "diarization_time": diarization_time,
            "whisper_time": whisper_time,
            "canary_time": canary_time,
            "fusion_time": fusion_time,
            "rtfx": duration / total_time if total_time > 0 else 0,
            "systems_used": ["whisper", "canary"] if canary_result else ["whisper"],
            "num_speaker_segments": len(speaker_segments),
        }

        # Create final result
        result = MeetingTranscription(
            segments=transcription_segments,
            full_text=final_text,
            speakers=unique_speakers,
            duration=duration,
            language=detected_language,
            metadata=metadata
        )

        logger.info(
            f"Transcription complete in {total_time:.2f}s "
            f"({metadata['rtfx']:.2f}x realtime)"
        )

        return result

    def _combine_asr_and_diarization(
        self,
        speaker_segments: List[SpeakerSegment],
        words: List,
        full_text: str
    ) -> List[TranscriptionSegment]:
        """
        Combine ASR words with speaker diarization

        Args:
            speaker_segments: Speaker segments from diarization
            words: Words from ASR (WordInfo or AlignedWord objects)
            full_text: Full transcription text

        Returns:
            List of TranscriptionSegment objects
        """
        if not speaker_segments:
            # No diarization, create single segment
            return [TranscriptionSegment(
                start=0.0,
                end=words[-1].end if words else 0.0,
                speaker="SPEAKER_00",
                text=full_text,
                confidence=sum(w.confidence for w in words) / len(words) if words else 0.0
            )]

        # Assign words to speakers based on timing
        transcription_segments = []
        current_segment = None

        for word in words:
            # Find speaker at this time
            word_time = (word.start + word.end) / 2
            speaker = self._find_speaker_at_time(speaker_segments, word_time)

            if speaker is None:
                # No speaker found, use previous or default
                speaker = current_segment.speaker if current_segment else "SPEAKER_00"

            # Check if we need to start a new segment
            if current_segment is None or current_segment.speaker != speaker:
                # Start new segment
                if current_segment:
                    transcription_segments.append(current_segment)

                current_segment = TranscriptionSegment(
                    start=word.start,
                    end=word.end,
                    speaker=speaker,
                    text=word.word,
                    confidence=word.confidence,
                    metadata={"word_count": 1}
                )
            else:
                # Add to current segment
                current_segment.text += " " + word.word
                current_segment.end = word.end
                current_segment.confidence = (
                    (current_segment.confidence * current_segment.metadata["word_count"] +
                     word.confidence) /
                    (current_segment.metadata["word_count"] + 1)
                )
                current_segment.metadata["word_count"] += 1

        # Add final segment
        if current_segment:
            transcription_segments.append(current_segment)

        return transcription_segments

    @staticmethod
    def _find_speaker_at_time(
        segments: List[SpeakerSegment],
        time: float
    ) -> Optional[str]:
        """Find which speaker is speaking at a given time"""
        for seg in segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        return None

    def save_results(
        self,
        transcription: MeetingTranscription,
        output_dir: str,
        base_name: str = "transcription",
        formats: List[str] = ["json", "txt", "srt"]
    ):
        """
        Save transcription results in multiple formats

        Args:
            transcription: MeetingTranscription object
            output_dir: Output directory
            base_name: Base filename
            formats: List of formats to export (json, txt, srt)
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for fmt in formats:
            if fmt == "json":
                json_path = output_path / f"{base_name}.json"
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(transcription.to_dict(), f, indent=2, ensure_ascii=False)
                logger.info(f"Saved JSON: {json_path}")

            elif fmt == "txt":
                txt_path = output_path / f"{base_name}.txt"
                transcription.export_txt(str(txt_path))
                logger.info(f"Saved TXT: {txt_path}")

            elif fmt == "srt":
                srt_path = output_path / f"{base_name}.srt"
                transcription.export_srt(str(srt_path))
                logger.info(f"Saved SRT: {srt_path}")
