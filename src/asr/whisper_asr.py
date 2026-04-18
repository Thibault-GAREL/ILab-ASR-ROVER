"""
Whisper ASR implementation using faster-whisper
Optimized for speed with minimal accuracy loss
"""

import logging
from typing import Optional, List, Dict, Any
import numpy as np

from faster_whisper import WhisperModel
import torchaudio

from .base_asr import BaseASR, ASRResult, WordInfo

logger = logging.getLogger(__name__)


class WhisperASR(BaseASR):
    """
    Whisper ASR using faster-whisper for optimized inference

    Supports all Whisper model sizes:
    - tiny, base, small, medium
    - large-v1, large-v2, large-v3 (recommended)
    """

    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "cuda",
        compute_type: str = "float16",
        **kwargs
    ):
        """
        Initialize Whisper ASR

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large-v3)
            device: Device to run on (cuda/cpu)
            compute_type: Computation precision (float16, int8, int8_float16)
            **kwargs: Additional parameters for WhisperModel
        """
        super().__init__(name=f"Whisper-{model_size}", device=device)

        self.model_size = model_size
        self.compute_type = compute_type

        logger.info(f"Loading Whisper model: {model_size} ({compute_type})")
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """Load the Whisper model"""
        try:
            # Filter kwargs to only include valid WhisperModel constructor parameters
            valid_params = {
                'device_index', 'inter_threads', 'intra_threads',
                'max_queued_batches', 'flash_attention', 'tensor_parallel', 'files'
            }
            model_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}

            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                **model_kwargs
            )
            logger.info(f"Whisper model loaded successfully: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: List[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        word_timestamps: bool = True,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe an audio file using Whisper

        Args:
            audio_path: Path to audio file
            language: Language code ("en", "fr", None for auto)
            beam_size: Beam size for decoding
            best_of: Number of candidates for temperature sampling
            temperature: Temperature values for sampling
            word_timestamps: Enable word-level timestamps
            vad_filter: Enable VAD filtering
            vad_parameters: VAD filter parameters
            **kwargs: Additional Whisper parameters

        Returns:
            ASRResult object
        """
        logger.info(f"Transcribing with Whisper: {audio_path}")

        try:
            # Set VAD parameters
            if vad_filter and vad_parameters is None:
                vad_parameters = {
                    "threshold": 0.5,
                    "min_speech_duration_ms": 250,
                    "min_silence_duration_ms": 2000,
                }

            # Transcribe
            segments, info = self.model.transcribe(
                audio_path,
                language=language,
                beam_size=beam_size,
                best_of=best_of,
                temperature=temperature,
                word_timestamps=word_timestamps,
                vad_filter=vad_filter,
                vad_parameters=vad_parameters,
                **kwargs
            )

            # Process segments
            all_words = []
            all_segments = []
            full_text_parts = []

            for segment in segments:
                segment_dict = {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text,
                    "avg_logprob": segment.avg_logprob,
                    "no_speech_prob": segment.no_speech_prob,
                }
                all_segments.append(segment_dict)
                full_text_parts.append(segment.text)

                # Extract word-level information
                if word_timestamps and segment.words:
                    for word in segment.words:
                        # Convert log probability to confidence (approximate)
                        confidence = self._logprob_to_confidence(word.probability)

                        word_info = WordInfo(
                            word=word.word.strip(),
                            start=word.start,
                            end=word.end,
                            confidence=confidence
                        )
                        all_words.append(word_info)

            # Combine full text
            full_text = " ".join(full_text_parts).strip()

            # Calculate overall confidence
            if all_words:
                avg_confidence = sum(w.confidence for w in all_words) / len(all_words)
            else:
                # Use average log probability from segments
                avg_logprob = np.mean([s["avg_logprob"] for s in all_segments])
                avg_confidence = self._logprob_to_confidence(avg_logprob)

            # Create result
            result = ASRResult(
                text=full_text,
                language=info.language,
                words=all_words,
                segments=all_segments,
                confidence=avg_confidence,
                metadata={
                    "model": self.model_size,
                    "language_probability": info.language_probability,
                    "duration": info.duration,
                    "num_segments": len(all_segments),
                }
            )

            logger.info(
                f"Whisper transcription complete: {len(all_words)} words, "
                f"{len(all_segments)} segments, language={info.language}"
            )

            return result

        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise

    def transcribe_segment(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe a specific segment of audio

        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds
            language: Language code
            **kwargs: Additional parameters

        Returns:
            ASRResult for the segment
        """
        logger.info(f"Transcribing segment [{start:.2f}s - {end:.2f}s]")

        try:
            # Load audio segment
            waveform, sample_rate = torchaudio.load(audio_path)

            # Extract segment
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]

            # Save temporary segment (Whisper needs file path)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
                torchaudio.save(tmp_path, segment_waveform, sample_rate)

            # Transcribe
            result = self.transcribe(tmp_path, language=language, **kwargs)

            # Adjust timestamps to absolute time
            for word in result.words:
                word.start += start
                word.end += start

            for segment in result.segments:
                segment["start"] += start
                segment["end"] += start

            # Cleanup
            import os
            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            raise

    @staticmethod
    def _logprob_to_confidence(logprob: float) -> float:
        """
        Convert log probability to confidence score [0, 1]

        Args:
            logprob: Log probability (negative value)

        Returns:
            Confidence score between 0 and 1
        """
        # exp(logprob) gives probability
        # Clip to reasonable range
        confidence = np.exp(logprob)
        return float(np.clip(confidence, 0.0, 1.0))

    def get_model_info(self) -> Dict[str, Any]:
        """Get Whisper model information"""
        info = super().get_model_info()
        info.update({
            "model_size": self.model_size,
            "compute_type": self.compute_type,
        })
        return info
