"""
Base class for ASR systems
Defines common interface for all ASR implementations
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class WordInfo:
    """Information about a single word in transcription"""
    word: str
    start: float
    end: float
    confidence: float
    speaker: Optional[str] = None

    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "speaker": self.speaker,
            "duration": self.duration()
        }


@dataclass
class ASRResult:
    """Result from an ASR system"""
    text: str
    language: str
    words: List[WordInfo] = field(default_factory=list)
    segments: List[Dict] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_text_in_range(self, start: float, end: float) -> str:
        """Extract text spoken in a time range"""
        words_in_range = [
            w.word for w in self.words
            if w.start >= start and w.end <= end
        ]
        return " ".join(words_in_range)

    def get_average_confidence(self) -> float:
        """Calculate average word confidence"""
        if not self.words:
            return self.confidence

        confidences = [w.confidence for w in self.words if w.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "language": self.language,
            "words": [w.to_dict() for w in self.words],
            "segments": self.segments,
            "confidence": self.confidence,
            "avg_word_confidence": self.get_average_confidence(),
            "metadata": self.metadata
        }


class BaseASR(ABC):
    """
    Abstract base class for ASR systems

    All ASR implementations must inherit from this class
    and implement the transcribe method.
    """

    def __init__(self, name: str, device: str = "cuda"):
        self.name = name
        self.device = device
        self.model = None
        logger.info(f"Initializing {name} ASR system on {device}")

    @abstractmethod
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe an audio file

        Args:
            audio_path: Path to audio file
            language: Language code (None for auto-detection)
            **kwargs: Additional parameters specific to the ASR system

        Returns:
            ASRResult object with transcription and metadata
        """
        pass

    @abstractmethod
    def transcribe_segment(
        self,
        audio_path: str,
        start: float,
        end: float,
        language: Optional[str] = None,
        **kwargs
    ) -> ASRResult:
        """
        Transcribe a specific segment of an audio file

        Args:
            audio_path: Path to audio file
            start: Start time in seconds
            end: End time in seconds
            language: Language code (None for auto-detection)
            **kwargs: Additional parameters

        Returns:
            ASRResult object for the segment
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            "name": self.name,
            "device": self.device,
            "loaded": self.model is not None
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', device='{self.device}')"
