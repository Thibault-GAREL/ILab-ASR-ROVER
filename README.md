# ASR ROVER: Multilingual Meeting Transcription

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

High-accuracy multilingual speech-to-text system for long meetings with speaker diarization, combining multiple state-of-the-art ASR systems using ROVER fusion.

## Features

- **Multi-System ASR**: Combines Whisper Large V3 (robustness) + NVIDIA Canary (accuracy)
- **ROVER Fusion**: Confidence-weighted voting to achieve best-in-class WER (~4.5-5.5%)
- **Speaker Diarization**: Pyannote Community-1 for accurate speaker identification
- **Multilingual**: Optimized for English and French, supports 99+ languages via Whisper
- **Production-Ready**: Modular architecture, extensive configuration options
- **Multiple Output Formats**: JSON, TXT, SRT subtitle format

## Architecture

```
┌─────────────────────────────────────────────┐
│ 1. DIARIZATION (Pyannote Community-1)      │
│    → Identify speakers and segments         │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│ 2. ASR SYSTEM 1: Whisper Large V3          │
│    → Multilingual robustness (99 langs)    │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│ 3. ASR SYSTEM 2: NVIDIA Canary             │
│    → High accuracy EN/FR/ES/DE             │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│ 4. ROVER FUSION                             │
│    → Confidence-weighted voting             │
│    → Word-level alignment                   │
│    → Best output selection                  │
└────────────┬────────────────────────────────┘
             │
┌────────────▼────────────────────────────────┐
│ 5. FINAL TRANSCRIPTION                      │
│    → Speaker-labeled segments               │
│    → Timestamps & confidence scores         │
└─────────────────────────────────────────────┘
```

## Performance Benchmarks

Based on 2025 state-of-the-art models:

| Component | Model | WER | Speed (RTFx) |
|-----------|-------|-----|--------------|
| ASR 1 | Whisper Large V3 | ~7-8% | 68x |
| ASR 2 | NVIDIA Canary Qwen 2.5B | ~5.6% | 418x |
| **ROVER Fusion** | **Combined** | **~4.5-5.5%** | **~240x** |
| Diarization | Pyannote Community-1 | ~10% DER | 2.5% RTFx |

## Installation

### Prerequisites

- Python 3.8+
- ~5GB disk space (for models)
- HuggingFace account (free - for Pyannote models)
- Optional: CUDA-capable GPU (8GB+ VRAM for faster processing)

### 🚀 Automated Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/Thibault-GAREL/ASR-Mixture_of_expert-ROVER.git
cd ASR-Mixture_of_expert-ROVER

# Run automated installation script
bash install_dependencies.sh

# Configure HuggingFace token
python setup_token.py

# Generate test audio and verify installation
python generate_test_audio.py
python test_diarization_fix.py
```

### Manual Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (Whisper-only, no conflicts)
pip install -r requirements-whisper-only.txt

# Clear Python cache (important!)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Configure HuggingFace token
python setup_token.py
```

### Full Install with Canary (Advanced)

⚠️ **Note:** NeMo has strict dependency requirements and may conflict with other packages.

```bash
pip install -r requirements-full.txt
```

For troubleshooting, see:
- **[QUICK_START.md](QUICK_START.md)** - Complete quick start guide
- **[FIX_CACHE_PYTHON.md](FIX_CACHE_PYTHON.md)** - Fix Python cache issues
- **[WINDOWS-GUIDE.md](WINDOWS-GUIDE.md)** - Windows-specific instructions

### HuggingFace Configuration

You must accept the licenses for these Pyannote models:
1. https://huggingface.co/pyannote/speaker-diarization-3.1
2. https://huggingface.co/pyannote/segmentation-3.0
3. https://huggingface.co/pyannote/speaker-diarization-community-1
4. https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
5. https://huggingface.co/pyannote/VoiceActivityDetection-PyanNet-ONNX

Then run:
```bash
python setup_token.py  # Interactive token setup
python diagnostic_hf.py  # Verify access to all models
```

## Quick Start

### Basic Usage

```python
from src.pipeline import MeetingTranscriptionPipeline

# Initialize pipeline
pipeline = MeetingTranscriptionPipeline()

# Transcribe meeting
result = pipeline.transcribe(
    audio_path="meeting.wav",
    language=None,  # Auto-detect
    num_speakers=None  # Auto-detect
)

# Access results
print(f"Duration: {result.duration}s")
print(f"Speakers: {result.speakers}")
print(f"Language: {result.language}")

for segment in result.segments:
    print(f"[{segment.speaker}]: {segment.text}")

# Save results
pipeline.save_results(
    transcription=result,
    output_dir="output",
    formats=["json", "txt", "srt"]
)
```

### Command-Line Interface

```bash
# Transcribe with auto-detection
python examples/cli_transcribe.py meeting.wav

# Specify language and speakers
python examples/cli_transcribe.py meeting.wav \
    --language fr \
    --num-speakers 3 \
    --output results/

# Use Whisper only (no Canary/ROVER)
python examples/cli_transcribe.py meeting.wav --whisper-only

# Custom output formats
python examples/cli_transcribe.py meeting.wav --formats json txt srt
```

## Configuration

### Using Config File

Edit `configs/config.yaml`:

```yaml
diarization:
  model_name: "pyannote/speaker-diarization-3.1"
  min_speakers: 2
  max_speakers: 10

whisper:
  model_size: "large-v3"
  device: "cuda"
  compute_type: "float16"
  beam_size: 5

canary:
  model_name: "nvidia/canary-1b"
  device: "cuda"

rover:
  voting_method: "confidence_weighted"
  confidence_weights:
    whisper: 1.0
    canary: 1.2  # Higher weight for better accuracy
```

### Programmatic Configuration

```python
pipeline = MeetingTranscriptionPipeline(
    diarizer_config={
        "model_name": "pyannote/speaker-diarization-3.1",
        "device": "cuda",
        "min_speakers": 2,
        "max_speakers": 5
    },
    whisper_config={
        "model_size": "large-v3",
        "compute_type": "float16"
    },
    canary_config={
        "model_name": "nvidia/canary-1b"
    },
    rover_config={
        "voting_method": "confidence_weighted",
        "confidence_weights": {"whisper": 1.0, "canary": 1.3}
    }
)
```

## Examples

See **[QUICK_START.md](QUICK_START.md)** for comprehensive usage guide.

### 1. Interactive Examples Menu
```bash
python examples/basic_usage.py
```

This provides an interactive menu with 6 different examples:
- Simple transcription
- Transcription with speaker identification
- Saving results to JSON
- Forcing specific language
- Batch processing multiple files
- Custom configuration

### 2. Command-Line Interface
```bash
# Basic transcription with auto-detection
python examples/cli_transcribe.py meeting.wav --whisper-only

# Specify language and speakers
python examples/cli_transcribe.py meeting.wav \
    --language fr \
    --num-speakers 3 \
    --whisper-only

# Custom output file
python examples/cli_transcribe.py meeting.wav --output results/transcript.json
```

### 3. Programmatic Usage
```python
from src.pipeline import TranscriptionPipeline

# Initialize (Whisper-only mode recommended)
pipeline = TranscriptionPipeline(
    config_path="configs/config.yaml",
    whisper_only=True
)

# Transcribe
result = pipeline.transcribe("meeting.wav")

# Access results
print(f"Text: {result.text}")
print(f"Language: {result.language}")
for seg in result.segments:
    print(f"[{seg.speaker}] {seg.text}")
```

## Output Formats

### JSON
```json
{
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "speaker": "SPEAKER_00",
      "text": "Hello, welcome to today's meeting.",
      "confidence": 0.95
    }
  ],
  "full_text": "...",
  "speakers": ["SPEAKER_00", "SPEAKER_01"],
  "duration": 3600.5,
  "language": "en"
}
```

### TXT
```
Meeting Transcription
Duration: 3600.50s
Language: en
Speakers: SPEAKER_00, SPEAKER_01
================================================================================

SPEAKER_00 [0.00s - 5.20s]:
Hello, welcome to today's meeting.

SPEAKER_01 [5.50s - 12.30s]:
Thank you for having me.
```

### SRT (Subtitles)
```
1
00:00:00,000 --> 00:00:05,200
[SPEAKER_00] Hello, welcome to today's meeting.

2
00:00:05,500 --> 00:00:12,300
[SPEAKER_01] Thank you for having me.
```

## Project Structure

```
ASR-Mixture_of_expert-ROVER/
├── configs/
│   └── config.yaml                 # Configuration file
├── src/
│   ├── asr/
│   │   ├── base_asr.py            # Base ASR interface
│   │   ├── whisper_asr.py         # Whisper implementation
│   │   └── canary_asr.py          # Canary implementation
│   ├── diarization/
│   │   └── pyannote_diarizer.py   # Speaker diarization
│   ├── rover/
│   │   └── rover_fusion.py        # ROVER fusion system
│   ├── utils/
│   │   ├── audio_utils.py         # Audio processing
│   │   └── config_loader.py       # Config management
│   └── pipeline.py                 # Main pipeline
├── examples/
│   ├── basic_transcription.py
│   ├── custom_config.py
│   ├── whisper_only.py
│   └── cli_transcribe.py
├── data/
│   ├── input/                      # Place audio files here
│   └── output/                     # Transcription outputs
├── requirements.txt
├── setup.py
└── README.md
```

## Advanced Usage

### Custom ROVER Voting

```python
from src.rover.rover_fusion import ROVERFusion

rover = ROVERFusion(
    voting_method="confidence_weighted",
    confidence_weights={
        "whisper": 1.0,
        "canary": 1.5  # Prioritize Canary
    },
    min_confidence_threshold=0.2,
    word_error_tolerance=0.15
)
```

### Speaker Diarization Only

```python
from src.diarization.pyannote_diarizer import PyannoteDiarizer

diarizer = PyannoteDiarizer(
    model_name="pyannote/speaker-diarization-3.1",
    device="cuda"
)

segments = diarizer.diarize("audio.wav", min_speakers=2, max_speakers=5)

for seg in segments:
    print(f"{seg.speaker}: {seg.start:.2f}s - {seg.end:.2f}s")
```

### ASR Only (No Diarization)

```python
from src.asr.whisper_asr import WhisperASR

whisper = WhisperASR(model_size="large-v3", device="cuda")
result = whisper.transcribe("audio.wav", language="fr")

print(result.text)
print(f"Confidence: {result.confidence}")
```

## Troubleshooting

**See [FIX_CACHE_PYTHON.md](FIX_CACHE_PYTHON.md) for detailed troubleshooting guide.**

### Issue: DiarizeOutput AttributeError
**Symptom**: `AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'`

**Solution**: Clear Python cache
```bash
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null
# Restart your terminal/IDE, then retry
```

### Issue: HuggingFace 401 Unauthorized
**Solution**: Accept model licenses and set token
```bash
python diagnostic_hf.py  # Identifies which licenses you're missing
python setup_token.py    # Set up your HF token
```

### Issue: CUDA Out of Memory
**Solution**: Use CPU or reduce model sizes
```yaml
# In config.yaml
whisper:
  device: "cpu"
  compute_type: "int8"
  model_size: "medium"  # Instead of "large-v3"
```

### Issue: Slow Processing on CPU
**Solution**:
- Use smaller Whisper model: `medium` instead of `large-v3`
- Reduce beam size: `beam_size: 1` (faster, slightly less accurate)
- Use Whisper-only mode (skip ROVER fusion)

### Issue: Dependency Conflicts with NeMo
**Solution**: Use Whisper-only mode
```bash
pip install -r requirements-whisper-only.txt
python examples/cli_transcribe.py audio.wav --whisper-only
```

## Helper Tools

The repository includes several diagnostic and setup tools:

| Tool | Purpose |
|------|---------|
| `setup_token.py` | Interactive HuggingFace token setup |
| `diagnostic_hf.py` | Diagnose HuggingFace access issues |
| `generate_test_audio.py` | Generate test WAV files |
| `test_diarization_fix.py` | Test diarization module |
| `install_dependencies.sh` | Automated dependency installation |
| `examples/basic_usage.py` | Interactive examples menu |

## Performance Tuning

### For Maximum Accuracy
```python
whisper_config={"model_size": "large-v3", "beam_size": 10}
rover_config={"confidence_weights": {"canary": 1.5}}
```

### For Maximum Speed
```python
whisper_config={"model_size": "medium", "compute_type": "int8"}
use_canary=False  # Skip ROVER fusion
```

### For Balanced Performance
```python
whisper_config={"model_size": "large-v3", "compute_type": "float16"}
canary_config={"decode_method": "greedy"}
```

## Citation

If you use this project in your research, please cite:

```bibtex
@software{asr_rover_2025,
  title={ASR ROVER: Multilingual Meeting Transcription with Multi-System Fusion},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/ASR-Mixture_of_expert-ROVER}
}
```

## References

- [Whisper (OpenAI)](https://github.com/openai/whisper)
- [faster-whisper](https://github.com/guillaumekln/faster-whisper)
- [NVIDIA Canary](https://developer.nvidia.com/blog/new-standard-for-speech-recognition-and-translation-from-the-nvidia-nemo-canary-model/)
- [Pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [ROVER Algorithm (NIST)](https://ieeexplore.ieee.org/document/659110/)
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/ASR-Mixture_of_expert-ROVER/issues)
- Documentation: See `examples/` directory
- Email: your.email@example.com

## Acknowledgments

- OpenAI for Whisper
- NVIDIA for Canary/NeMo
- Pyannote team for speaker diarization
- HuggingFace for model hosting and Open ASR Leaderboard
