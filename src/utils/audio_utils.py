"""Audio processing utilities"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import torch
import torchaudio
import numpy as np

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Utilities for audio preprocessing"""

    def __init__(
        self,
        target_sample_rate: int = 16000,
        normalize: bool = True,
        trim_silence: bool = False,
        silence_threshold_db: float = -40
    ):
        """
        Initialize audio processor

        Args:
            target_sample_rate: Target sample rate (16kHz for most ASR)
            normalize: Normalize audio amplitude
            trim_silence: Remove leading/trailing silence
            silence_threshold_db: Silence threshold in dB
        """
        self.target_sample_rate = target_sample_rate
        self.normalize = normalize
        self.trim_silence = trim_silence
        self.silence_threshold_db = silence_threshold_db

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """
        Load audio file

        Args:
            audio_path: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)
        """
        if not Path(audio_path).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Loading audio: {audio_path}")
        waveform, sample_rate = torchaudio.load(audio_path)

        return waveform, sample_rate

    def preprocess(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> Tuple[torch.Tensor, int]:
        """
        Preprocess audio waveform

        Args:
            waveform: Audio waveform tensor
            sample_rate: Original sample rate

        Returns:
            Tuple of (processed_waveform, sample_rate)
        """
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            logger.debug("Converting stereo to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            logger.debug(f"Resampling from {sample_rate}Hz to {self.target_sample_rate}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)
            sample_rate = self.target_sample_rate

        # Normalize
        if self.normalize:
            waveform = self._normalize_audio(waveform)

        # Trim silence
        if self.trim_silence:
            waveform = self._trim_silence(waveform, sample_rate)

        return waveform, sample_rate

    def save_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        output_path: str,
        format: Optional[str] = None
    ):
        """
        Save audio to file

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            output_path: Output file path
            format: Audio format (None for auto-detect from extension)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        torchaudio.save(
            str(output_path),
            waveform,
            sample_rate,
            format=format
        )
        logger.info(f"Audio saved: {output_path}")

    @staticmethod
    def _normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio to [-1, 1] range"""
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            return waveform / max_val
        return waveform

    def _trim_silence(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> torch.Tensor:
        """
        Trim silence from beginning and end

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            Trimmed waveform
        """
        # Convert dB threshold to amplitude
        threshold = 10 ** (self.silence_threshold_db / 20)

        # Find non-silent regions
        magnitude = torch.abs(waveform)
        non_silent = magnitude > threshold

        # Find first and last non-silent sample
        non_silent_indices = torch.where(non_silent[0])[0]

        if len(non_silent_indices) == 0:
            # All silence, return as is
            return waveform

        start_idx = non_silent_indices[0].item()
        end_idx = non_silent_indices[-1].item()

        return waveform[:, start_idx:end_idx+1]

    def get_duration(
        self,
        waveform: torch.Tensor,
        sample_rate: int
    ) -> float:
        """
        Get audio duration in seconds

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate

        Returns:
            Duration in seconds
        """
        return waveform.shape[1] / sample_rate

    def split_audio(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
        segment_length: float,
        overlap: float = 0.0
    ) -> list:
        """
        Split audio into segments

        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            segment_length: Segment length in seconds
            overlap: Overlap between segments in seconds

        Returns:
            List of (segment_waveform, start_time, end_time) tuples
        """
        segment_samples = int(segment_length * sample_rate)
        overlap_samples = int(overlap * sample_rate)
        step_samples = segment_samples - overlap_samples

        segments = []
        start = 0

        while start < waveform.shape[1]:
            end = min(start + segment_samples, waveform.shape[1])
            segment = waveform[:, start:end]

            start_time = start / sample_rate
            end_time = end / sample_rate

            segments.append((segment, start_time, end_time))

            if end >= waveform.shape[1]:
                break

            start += step_samples

        logger.info(f"Split audio into {len(segments)} segments")
        return segments

    @staticmethod
    def convert_format(
        input_path: str,
        output_path: str,
        target_format: str = "wav",
        target_sample_rate: int = 16000
    ):
        """
        Convert audio file format

        Args:
            input_path: Input audio file
            output_path: Output audio file
            target_format: Target format (wav, mp3, flac, etc.)
            target_sample_rate: Target sample rate
        """
        waveform, sample_rate = torchaudio.load(input_path)

        # Resample if needed
        if sample_rate != target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=target_sample_rate
            )
            waveform = resampler(waveform)

        # Save in target format
        torchaudio.save(
            output_path,
            waveform,
            target_sample_rate,
            format=target_format
        )
        logger.info(f"Converted {input_path} to {output_path}")
