"""Configuration loading utilities"""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to config file (default: configs/config.yaml)

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Default config path
        project_root = Path(__file__).parent.parent.parent
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}. Using defaults.")
        return get_default_config()

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override with environment variables
    config = _apply_env_overrides(config)

    return config


def get_default_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "general": {
            "device": "cuda",
            "num_threads": 4,
            "output_format": "json",
            "log_level": "INFO"
        },
        "diarization": {
            "model_name": "pyannote/speaker-diarization-3.1",
            "min_speakers": None,
            "max_speakers": None,
            "hf_token": None
        },
        "whisper": {
            "model_size": "large-v3",
            "device": "cuda",
            "compute_type": "float16",
            "language": None,
            "beam_size": 5,
            "word_timestamps": True,
            "vad_filter": True
        },
        "canary": {
            "model_name": "nvidia/canary-1b",
            "device": "cuda",
            "decode_method": "greedy",
            "beam_width": 32
        },
        "rover": {
            "voting_method": "confidence_weighted",
            "confidence_weights": {
                "whisper": 1.0,
                "canary": 1.2
            },
            "min_confidence_threshold": 0.3,
            "word_error_tolerance": 0.15,
            "tie_breaking": "highest_confidence"
        },
        "audio": {
            "sample_rate": 16000,
            "normalize": True,
            "trim_silence": True,
            "silence_threshold": -40
        },
        "output": {
            "include_timestamps": True,
            "include_confidence_scores": True,
            "include_speaker_labels": True,
            "save_intermediate_results": False
        }
    }


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config

    Args:
        config: Base configuration

    Returns:
        Updated configuration
    """
    # Device override
    if "DEVICE" in os.environ:
        device = os.environ["DEVICE"]
        if "general" in config:
            config["general"]["device"] = device
        if "whisper" in config:
            config["whisper"]["device"] = device
        if "canary" in config:
            config["canary"]["device"] = device

    # HuggingFace token
    if "HF_TOKEN" in os.environ:
        if "diarization" in config:
            config["diarization"]["hf_token"] = os.environ["HF_TOKEN"]

    # Log level
    if "LOG_LEVEL" in os.environ:
        if "general" in config:
            config["general"]["log_level"] = os.environ["LOG_LEVEL"]

    return config


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)

    logger.info(f"Config saved to: {output_path}")


def setup_logging(config: Dict[str, Any]):
    """
    Setup logging based on config

    Args:
        config: Configuration dictionary
    """
    log_level = config.get("general", {}).get("log_level", "INFO")

    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info(f"Logging configured: level={log_level}")
