"""
ROVER (Recognizer Output Voting Error Reduction)
Fuses multiple ASR system outputs using voting and confidence weighting
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import difflib

from ..asr.base_asr import ASRResult, WordInfo

logger = logging.getLogger(__name__)


@dataclass
class AlignedWord:
    """A word aligned across multiple ASR systems"""
    word: str
    start: float
    end: float
    confidence: float
    source_system: str
    votes: int = 1
    alternative_words: List[Tuple[str, float, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "source_system": self.source_system,
            "votes": self.votes,
            "alternatives": [
                {"word": w, "confidence": c, "source": s}
                for w, c, s in self.alternative_words
            ]
        }


@dataclass
class FusionResult:
    """Result from ROVER fusion"""
    text: str
    words: List[AlignedWord]
    confidence: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "words": [w.to_dict() for w in self.words],
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class ROVERFusion:
    """
    ROVER system for fusing multiple ASR outputs

    Implements:
    - Word-level alignment
    - Confidence-weighted voting
    - Multiple voting strategies
    """

    VOTING_METHODS = ["majority", "confidence_weighted", "rank_weighted"]

    def __init__(
        self,
        voting_method: str = "confidence_weighted",
        confidence_weights: Optional[Dict[str, float]] = None,
        min_confidence_threshold: float = 0.3,
        word_error_tolerance: float = 0.15,
        tie_breaking: str = "highest_confidence",
    ):
        """
        Initialize ROVER fusion system

        Args:
            voting_method: Voting method (majority, confidence_weighted, rank_weighted)
            confidence_weights: System-specific weights (e.g., {"whisper": 1.0, "canary": 1.2})
            min_confidence_threshold: Minimum confidence to consider a word
            word_error_tolerance: Tolerance for fuzzy word matching (0-1)
            tie_breaking: Tie breaking strategy (highest_confidence, first_system)
        """
        if voting_method not in self.VOTING_METHODS:
            raise ValueError(f"Invalid voting method: {voting_method}")

        self.voting_method = voting_method
        self.confidence_weights = confidence_weights or {}
        self.min_confidence_threshold = min_confidence_threshold
        self.word_error_tolerance = word_error_tolerance
        self.tie_breaking = tie_breaking

        logger.info(f"ROVER initialized with {voting_method} voting")

    def fuse(
        self,
        asr_results: Dict[str, ASRResult],
        use_word_alignment: bool = True
    ) -> FusionResult:
        """
        Fuse multiple ASR results using ROVER

        Args:
            asr_results: Dictionary mapping system name to ASRResult
            use_word_alignment: Use word-level alignment (recommended)

        Returns:
            FusionResult with fused transcription
        """
        if len(asr_results) < 2:
            raise ValueError("ROVER requires at least 2 ASR systems")

        logger.info(f"Fusing {len(asr_results)} ASR systems: {list(asr_results.keys())}")

        if use_word_alignment:
            return self._fuse_word_level(asr_results)
        else:
            return self._fuse_text_level(asr_results)

    def _fuse_word_level(self, asr_results: Dict[str, ASRResult]) -> FusionResult:
        """
        Word-level ROVER fusion with timestamp alignment

        Args:
            asr_results: Dictionary of ASR results

        Returns:
            Fused result
        """
        # Build time-aligned word grid
        word_grid = self._build_word_grid(asr_results)

        # Vote on each time slot
        fused_words = []
        for time_slot, candidates in sorted(word_grid.items()):
            selected_word = self._vote_on_candidates(candidates)
            if selected_word:
                fused_words.append(selected_word)

        # Merge consecutive words and build text
        merged_words = self._merge_consecutive_words(fused_words)
        text = " ".join(w.word for w in merged_words)

        # Calculate overall confidence
        if merged_words:
            avg_confidence = sum(w.confidence for w in merged_words) / len(merged_words)
        else:
            avg_confidence = 0.0

        # Create metadata
        metadata = {
            "num_systems": len(asr_results),
            "systems": list(asr_results.keys()),
            "voting_method": self.voting_method,
            "num_words": len(merged_words),
            "num_time_slots": len(word_grid),
        }

        logger.info(
            f"ROVER fusion complete: {len(merged_words)} words, "
            f"confidence={avg_confidence:.3f}"
        )

        return FusionResult(
            text=text,
            words=merged_words,
            confidence=avg_confidence,
            metadata=metadata
        )

    def _build_word_grid(
        self,
        asr_results: Dict[str, ASRResult]
    ) -> Dict[float, List[Tuple[str, WordInfo, str]]]:
        """
        Build a time-aligned grid of words from multiple systems

        Args:
            asr_results: ASR results from different systems

        Returns:
            Dictionary mapping time slots to candidate words
        """
        # Collect all word timings
        all_timings = set()
        for result in asr_results.values():
            for word in result.words:
                all_timings.add(word.start)
                all_timings.add(word.end)

        # Sort and create time bins
        sorted_timings = sorted(all_timings)

        # Map each word to time bins
        word_grid = defaultdict(list)

        for system_name, result in asr_results.items():
            for word in result.words:
                # Find the appropriate time slot (use start time)
                time_key = self._find_nearest_time_slot(word.start, sorted_timings)
                word_grid[time_key].append((system_name, word, word.word))

        return word_grid

    def _find_nearest_time_slot(self, time: float, time_slots: List[float]) -> float:
        """Find the nearest time slot for a given time"""
        if not time_slots:
            return time

        # Binary search for nearest
        closest = min(time_slots, key=lambda t: abs(t - time))
        return closest

    def _vote_on_candidates(
        self,
        candidates: List[Tuple[str, WordInfo, str]]
    ) -> Optional[AlignedWord]:
        """
        Vote on word candidates from different systems

        Args:
            candidates: List of (system_name, WordInfo, word_text) tuples

        Returns:
            Selected AlignedWord or None
        """
        if not candidates:
            return None

        # Filter by confidence threshold
        valid_candidates = [
            (sys, word, text) for sys, word, text in candidates
            if word.confidence >= self.min_confidence_threshold
        ]

        if not valid_candidates:
            # Use all candidates if none meet threshold
            valid_candidates = candidates

        # Group similar words (fuzzy matching)
        word_groups = self._group_similar_words(valid_candidates)

        # Vote based on method
        if self.voting_method == "majority":
            selected = self._majority_vote(word_groups)
        elif self.voting_method == "confidence_weighted":
            selected = self._confidence_weighted_vote(word_groups)
        elif self.voting_method == "rank_weighted":
            selected = self._rank_weighted_vote(word_groups)
        else:
            selected = self._confidence_weighted_vote(word_groups)

        return selected

    def _group_similar_words(
        self,
        candidates: List[Tuple[str, WordInfo, str]]
    ) -> Dict[str, List[Tuple[str, WordInfo]]]:
        """
        Group similar words together (handles spelling variations)

        Args:
            candidates: List of candidate words

        Returns:
            Dictionary mapping normalized word to list of (system, WordInfo)
        """
        groups = defaultdict(list)

        for system, word_info, word_text in candidates:
            # Normalize word
            normalized = word_text.lower().strip()

            # Find matching group (fuzzy)
            matched = False
            for existing_word in list(groups.keys()):
                similarity = difflib.SequenceMatcher(
                    None, normalized, existing_word
                ).ratio()

                if similarity >= (1.0 - self.word_error_tolerance):
                    # Close enough, group together
                    groups[existing_word].append((system, word_info))
                    matched = True
                    break

            if not matched:
                # Create new group
                groups[normalized].append((system, word_info))

        return groups

    def _majority_vote(
        self,
        word_groups: Dict[str, List[Tuple[str, WordInfo]]]
    ) -> Optional[AlignedWord]:
        """Simple majority voting"""
        if not word_groups:
            return None

        # Find group with most votes
        max_votes = 0
        winning_word = None
        winning_group = None

        for word, group in word_groups.items():
            votes = len(group)
            if votes > max_votes:
                max_votes = votes
                winning_word = word
                winning_group = group

        # Handle tie
        if sum(1 for g in word_groups.values() if len(g) == max_votes) > 1:
            if self.tie_breaking == "highest_confidence":
                return self._confidence_weighted_vote(word_groups)

        # Select representative from winning group
        return self._create_aligned_word(winning_word, winning_group, max_votes, word_groups)

    def _confidence_weighted_vote(
        self,
        word_groups: Dict[str, List[Tuple[str, WordInfo]]]
    ) -> Optional[AlignedWord]:
        """Confidence-weighted voting (recommended)"""
        if not word_groups:
            return None

        # Calculate weighted score for each word
        max_score = 0
        winning_word = None
        winning_group = None

        for word, group in word_groups.items():
            score = 0
            for system, word_info in group:
                # Apply system weight
                weight = self.confidence_weights.get(system, 1.0)
                score += word_info.confidence * weight

            if score > max_score:
                max_score = score
                winning_word = word
                winning_group = group

        votes = len(winning_group)
        return self._create_aligned_word(winning_word, winning_group, votes, word_groups)

    def _rank_weighted_vote(
        self,
        word_groups: Dict[str, List[Tuple[str, WordInfo]]]
    ) -> Optional[AlignedWord]:
        """Rank-weighted voting (system ranks as weights)"""
        # Similar to confidence-weighted but uses rank-based weights
        # For simplicity, fall back to confidence-weighted
        return self._confidence_weighted_vote(word_groups)

    def _create_aligned_word(
        self,
        word: str,
        group: List[Tuple[str, WordInfo]],
        votes: int,
        all_groups: Dict[str, List[Tuple[str, WordInfo]]]
    ) -> AlignedWord:
        """
        Create an AlignedWord from voting result

        Args:
            word: Selected word text
            group: List of (system, WordInfo) that voted for this word
            votes: Number of votes
            all_groups: All word groups (for alternatives)

        Returns:
            AlignedWord object
        """
        # Calculate average timing and confidence
        avg_start = sum(w.start for _, w in group) / len(group)
        avg_end = sum(w.end for _, w in group) / len(group)
        avg_confidence = sum(w.confidence for _, w in group) / len(group)

        # Select source system (highest confidence)
        source_system = max(group, key=lambda x: x[1].confidence)[0]

        # Collect alternatives
        alternatives = []
        for alt_word, alt_group in all_groups.items():
            if alt_word != word:
                for sys, word_info in alt_group:
                    alternatives.append((word_info.word, word_info.confidence, sys))

        return AlignedWord(
            word=word,
            start=avg_start,
            end=avg_end,
            confidence=avg_confidence,
            source_system=source_system,
            votes=votes,
            alternative_words=alternatives
        )

    def _merge_consecutive_words(
        self,
        words: List[AlignedWord],
        time_gap_threshold: float = 0.1
    ) -> List[AlignedWord]:
        """
        Merge duplicate consecutive words

        Args:
            words: List of aligned words
            time_gap_threshold: Maximum time gap to consider consecutive

        Returns:
            List of merged words
        """
        if not words:
            return []

        merged = [words[0]]

        for current in words[1:]:
            previous = merged[-1]

            # Check if same word and close in time
            if (current.word.lower() == previous.word.lower() and
                current.start - previous.end <= time_gap_threshold):
                # Merge: extend end time, average confidence
                previous.end = current.end
                previous.confidence = (previous.confidence + current.confidence) / 2
                previous.votes += current.votes
            else:
                merged.append(current)

        return merged

    def _fuse_text_level(self, asr_results: Dict[str, ASRResult]) -> FusionResult:
        """
        Simple text-level fusion (fallback if no word timestamps)

        Args:
            asr_results: ASR results

        Returns:
            Fused result
        """
        # Simple voting on full text
        # This is a simplified implementation
        logger.warning("Using text-level fusion (word-level preferred)")

        # Select result with highest confidence
        best_system = max(
            asr_results.items(),
            key=lambda x: x[1].confidence * self.confidence_weights.get(x[0], 1.0)
        )

        return FusionResult(
            text=best_system[1].text,
            words=[],
            confidence=best_system[1].confidence,
            metadata={
                "num_systems": len(asr_results),
                "selected_system": best_system[0],
                "voting_method": "text_level_fallback"
            }
        )
