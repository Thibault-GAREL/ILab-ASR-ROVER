"""
NVIDIA Canary ASR implementation using NeMo
High accuracy multilingual ASR for EN, FR, ES, DE
"""

import logging
from typing import Optional, Dict, Any, List
import os

try:
    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models import EncDecMultiTaskModel
    NEMO_AVAILABLE = True
except ImportError:
    NEMO_AVAILABLE = False
    logging.warning("NeMo not available. Install with: pip install nemo_toolkit[asr]")

import torchaudio
import torch

from .base_asr import BaseASR, ASRResult, WordInfo

logger = logging.getLogger(__name__)


class CanaryASR(BaseASR):
    """
    NVIDIA Canary ASR using NeMo toolkit

    Supports:
    - nvidia/canary-1b
    - Future: canary-qwen-2.5b (when available)

    Features:
    - High accuracy multilingual ASR
    - Word-level timestamps and confidence
    - Support for EN, FR, ES, DE
    """

    def __init__(
        self,
        model_name: str = "nvidia/canary-1b",
        device: str = "cuda",
        decode_method: str = "greedy",
        beam_width: int = 32,
        **kwargs
    ):
        """
        Initialize Canary ASR

        Args:
            model_name: Model identifier (nvidia/canary-1b)
            device: Device to run on (cuda/cpu)
            decode_method: Decoding method (greedy/beam)
            beam_width: Beam width for beam search
            **kwargs: Additional NeMo parameters
        """
        if not NEMO_AVAILABLE:
            raise ImportError(
                "NeMo toolkit is required for Canary ASR. "
                "Install with: pip install nemo_toolkit[asr]"
            )

        super().__init__(name=f"Canary-{model_name.split('/')[-1]}", device=device)

        self.model_name = model_name
        self.decode_method = decode_method
        self.beam_width = beam_width

        logger.info(f"Loading Canary model: {model_name}")
        self._load_model(**kwargs)

    def _load_model(self, **kwargs):
        """Load the Canary model"""
        try:
            # Filter kwargs - from_pretrained() only accepts specific parameters
            # Remove runtime parameters that should not be passed to model loading
            load_kwargs = {k: v for k, v in kwargs.items()
                          if k not in ['batch_size', 'language', 'decode_method',
                                      'beam_width', 'preserve_alignment',
                                      'compute_timestamps', 'compute_word_confidence']}

            # Load pretrained Canary model
            self.model = EncDecMultiTaskModel.from_pretrained(
                model_name=self.model_name,
                map_location=self.device,
                **load_kwargs
            )

            # Set to evaluation mode
            self.model.eval()

            # Move to device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info(f"Canary model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                logger.info("Canary model loaded on CPU")

            # Configure decoding
            self.model.change_decoding_strategy(
                decoder_type=self.decode_method,
                beam_width=self.beam_width if self.decode_method == "beam" else None
            )

        except Exception as e:
            logger.error(f"Failed to load Canary model: {e}")
            logger.info(
                "Note: Canary models may require NGC API key. "
                "Set NGC_API_KEY environment variable or download manually."
            )
            raise

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "asr",
        compute_timestamps: bool = True,
        compute_word_confidence: bool = True,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe an audio file using Canary

        Args:
            audio_path: Path to audio file
            language: Language code ("en", "fr", "es", "de", None for auto)
            task: Task type ("asr" for transcription)
            compute_timestamps: Enable word timestamps
            compute_word_confidence: Enable word confidence scores
            **kwargs: Additional parameters

        Returns:
            ASRResult object
        """
        logger.info(f"Transcribing with Canary: {audio_path}")

        try:
            # Prepare input
            audio_files = [audio_path]

            # Configure task
            # Canary uses task tokens: <|en|>, <|fr|>, <|transcribe|>, etc.
            if language:
                lang_token = f"<|{language}|>"
            else:
                lang_token = None  # Auto-detect

            # Transcribe
            with torch.no_grad():
                transcriptions = self.model.transcribe(
                    paths2audio_files=audio_files,
                    batch_size=1,
                    return_hypotheses=True,
                    **kwargs
                )

            # Extract hypothesis
            if not transcriptions or len(transcriptions) == 0:
                raise ValueError("No transcription returned from Canary")

            hypothesis = transcriptions[0]

            # Extract text
            text = hypothesis.text if hasattr(hypothesis, 'text') else str(hypothesis)

            # Extract word-level information
            words = []
            if compute_timestamps and hasattr(hypothesis, 'words'):
                for word_info in hypothesis.words:
                    word = WordInfo(
                        word=word_info.word if hasattr(word_info, 'word') else str(word_info),
                        start=word_info.start_offset if hasattr(word_info, 'start_offset') else 0.0,
                        end=word_info.end_offset if hasattr(word_info, 'end_offset') else 0.0,
                        confidence=word_info.confidence if hasattr(word_info, 'confidence') else 1.0,
                    )
                    words.append(word)

            # Calculate overall confidence
            if words:
                avg_confidence = sum(w.confidence for w in words) / len(words)
            elif hasattr(hypothesis, 'score'):
                avg_confidence = float(hypothesis.score)
            else:
                avg_confidence = 1.0

            # Detect language (if not specified)
            detected_language = language if language else self._detect_language(text)

            # Create result
            result = ASRResult(
                text=text.strip(),
                language=detected_language,
                words=words,
                segments=[{
                    "start": 0.0,
                    "end": words[-1].end if words else 0.0,
                    "text": text.strip()
                }],
                confidence=avg_confidence,
                metadata={
                    "model": self.model_name,
                    "decode_method": self.decode_method,
                    "num_words": len(words),
                }
            )

            logger.info(
                f"Canary transcription complete: {len(words)} words, "
                f"confidence={avg_confidence:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Canary transcription failed: {e}")
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

            # Save temporary segment
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
            os.unlink(tmp_path)

            return result

        except Exception as e:
            logger.error(f"Segment transcription failed: {e}")
            raise

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        Simple language detection based on common words
        (Fallback when model doesn't provide language info)

        Args:
            text: Transcribed text

        Returns:
            Detected language code
        """
        text_lower = text.lower()

        # French indicators
        french_words = ["le", "la", "les", "de", "et", "un", "une", "des", "je", "tu", "il", "elle"]
        french_score = sum(1 for word in french_words if f" {word} " in f" {text_lower} ")

        # English indicators
        english_words = ["the", "and", "is", "are", "to", "of", "in", "that", "it", "for"]
        english_score = sum(1 for word in english_words if f" {word} " in f" {text_lower} ")

        if french_score > english_score:
            return "fr"
        else:
            return "en"

    def get_model_info(self) -> Dict[str, Any]:
        """Get Canary model information"""
        info = super().get_model_info()
        info.update({
            "model_name": self.model_name,
            "decode_method": self.decode_method,
            "beam_width": self.beam_width,
        })
        return info
