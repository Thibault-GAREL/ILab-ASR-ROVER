from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="asr-rover-multilingual",
    version="1.0.0",
    author="Your Name",
    description="Multilingual ASR with ROVER fusion and speaker diarization for long meetings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/ASR-Mixture_of_expert-ROVER",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy>=1.24.0",
        "faster-whisper>=0.10.0",
        "nemo_toolkit[asr]>=1.22.0",
        "pyannote.audio>=3.1.0",
        "soundfile>=0.12.1",
        "librosa>=0.10.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "black>=23.0.0", "flake8>=6.0.0"],
    },
)
