"""
Unit tests for ROVER fusion system
"""

import pytest
from src.rover.rover_fusion import ROVERFusion, AlignedWord
from src.asr.base_asr import ASRResult, WordInfo


def test_rover_initialization():
    """Test ROVER initialization with different configurations"""
    rover = ROVERFusion(
        voting_method="confidence_weighted",
        confidence_weights={"whisper": 1.0, "canary": 1.2}
    )
    assert rover.voting_method == "confidence_weighted"
    assert rover.confidence_weights["canary"] == 1.2


def test_rover_fusion_basic():
    """Test basic ROVER fusion with two ASR results"""
    # Create mock ASR results
    words1 = [
        WordInfo("hello", 0.0, 0.5, 0.9),
        WordInfo("world", 0.5, 1.0, 0.85)
    ]
    words2 = [
        WordInfo("hello", 0.0, 0.5, 0.95),
        WordInfo("world", 0.5, 1.0, 0.9)
    ]

    result1 = ASRResult(
        text="hello world",
        language="en",
        words=words1,
        confidence=0.875
    )
    result2 = ASRResult(
        text="hello world",
        language="en",
        words=words2,
        confidence=0.925
    )

    # Fuse results
    rover = ROVERFusion(voting_method="confidence_weighted")
    fusion_result = rover.fuse({
        "whisper": result1,
        "canary": result2
    })

    assert fusion_result.text == "hello world"
    assert fusion_result.confidence > 0.8
    assert len(fusion_result.words) > 0


def test_rover_invalid_voting_method():
    """Test that invalid voting method raises error"""
    with pytest.raises(ValueError):
        ROVERFusion(voting_method="invalid_method")


if __name__ == "__main__":
    pytest.main([__file__])
