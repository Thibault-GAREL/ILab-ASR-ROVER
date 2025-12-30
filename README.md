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
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- HuggingFace account (for Pyannote models)

### Quick Install (Recommended - Whisper Only)

```bash
# Clone repository
git clone https://github.com/yourusername/ASR-Mixture_of_expert-ROVER.git
cd ASR-Mixture_of_expert-ROVER

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies (Whisper-only, no conflicts)
pip install -r requirements-whisper-only.txt
```

### Full Install with Canary (Advanced)

⚠️ **Note:** NeMo has strict dependency requirements. See [INSTALL.md](INSTALL.md) for detailed instructions.

```bash
# Sequential installation to avoid conflicts
pip install numpy==1.23.5
pip install nemo_toolkit[asr]==1.23.0
pip install -r requirements-full.txt
```

**See [INSTALL.md](INSTALL.md) for complete installation guide and troubleshooting.**

### Configuration

1. Copy environment template:
```bash
cp .env.example .env
```

2. Set your HuggingFace token in `.env`:
```bash
HF_TOKEN=your_huggingface_token_here
```

Get your token at: https://huggingface.co/settings/tokens

3. (Optional) For NVIDIA Canary, set NGC API key:
```bash
NGC_API_KEY=your_ngc_api_key_here
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

### 1. Basic Transcription
```python
python examples/basic_transcription.py
```

### 2. Custom Configuration
```python
python examples/custom_config.py
```

### 3. Whisper Only (No Canary)
```python
python examples/whisper_only.py
```

### 4. Command-Line Interface
```bash
python examples/cli_transcribe.py audio.wav --language en --num-speakers 2
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

### Issue: HuggingFace Token Error
**Solution**: Set `HF_TOKEN` environment variable or in `.env` file

### Issue: CUDA Out of Memory
**Solution**: Reduce model sizes or use CPU:
```python
whisper_config={"model_size": "medium", "compute_type": "int8"}
```

### Issue: Canary Model Not Loading
**Solution**: Use Whisper-only mode:
```python
pipeline = MeetingTranscriptionPipeline(use_canary=False)
```

### Issue: Slow Processing
**Solution**:
- Use GPU instead of CPU
- Reduce Whisper model size (`medium` instead of `large-v3`)
- Disable VAD filter for shorter files

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
