"""
Script de test simple pour vérifier l'installation
Pas besoin de fichier audio
"""

import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

print("=" * 60)
print("Test d'Installation ASR ROVER")
print("=" * 60)

# Test 1: Imports
print("\n1. Test des imports...")
try:
    import torch
    import torchaudio
    import numpy as np
    from faster_whisper import WhisperModel
    print("   ✓ PyTorch, torchaudio, numpy, faster-whisper")
except ImportError as e:
    print(f"   ✗ Erreur import: {e}")
    sys.exit(1)

try:
    from pyannote.audio import Pipeline
    print("   ✓ Pyannote.audio")
except ImportError as e:
    print(f"   ✗ Erreur pyannote: {e}")
    print("   → Installez avec: pip install pyannote.audio")
    sys.exit(1)

# Test 2: Device
print("\n2. Test du device...")
if torch.cuda.is_available():
    print(f"   ✓ CUDA disponible: {torch.cuda.get_device_name()}")
    device = "cuda"
else:
    print("   ⚠ CUDA non disponible, utilisation du CPU")
    device = "cpu"

# Test 3: Classes du projet
print("\n3. Test des modules du projet...")
try:
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.asr.whisper_asr import WhisperASR
    from src.diarization.pyannote_diarizer import PyannoteDiarizer
    from src.pipeline import MeetingTranscriptionPipeline
    print("   ✓ Modules du projet importés")
except ImportError as e:
    print(f"   ✗ Erreur: {e}")
    sys.exit(1)

# Test 4: Initialisation Whisper
print("\n4. Test initialisation Whisper...")
try:
    model_size = "tiny" if device == "cpu" else "base"
    compute_type = "int8" if device == "cpu" else "float16"

    print(f"   Chargement du modèle {model_size} ({compute_type})...")
    whisper = WhisperASR(
        model_size=model_size,
        device=device,
        compute_type=compute_type
    )
    print("   ✓ Whisper initialisé avec succès")
except Exception as e:
    print(f"   ✗ Erreur Whisper: {e}")
    sys.exit(1)

# Test 5: Check HuggingFace token
print("\n5. Vérification token HuggingFace...")
import os

# Try to load .env file if dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("   ⚠ python-dotenv pas installé (optionnel)")
    print("   → Installez avec: pip install python-dotenv")

hf_token = os.getenv("HF_TOKEN")

if hf_token:
    if hf_token.startswith("hf_"):
        print(f"   ✓ Token HF trouvé: {hf_token[:10]}...")
    else:
        print(f"   ⚠ Token trouvé mais format invalide (doit commencer par 'hf_')")
else:
    print("   ⚠ Pas de token HF dans .env")
    print("   → Créez un token sur: https://huggingface.co/settings/tokens")
    print("   → Ajoutez dans .env: HF_TOKEN=hf_...")

# Test 6: Test Diarizer (si token disponible)
if hf_token and hf_token.startswith("hf_"):
    print("\n6. Test initialisation Diarizer...")
    try:
        diarizer = PyannoteDiarizer(
            model_name="pyannote/speaker-diarization-3.1",
            device=device,
            hf_token=hf_token
        )
        print("   ✓ Diarizer initialisé avec succès")
    except Exception as e:
        print(f"   ⚠ Erreur Diarizer: {e}")
        print("   → Vérifiez que vous avez accepté les conditions sur:")
        print("   → https://huggingface.co/pyannote/speaker-diarization-3.1")
else:
    print("\n6. Test Diarizer ignoré (pas de token HF valide)")

# Résumé
print("\n" + "=" * 60)
print("RÉSUMÉ DES TESTS")
print("=" * 60)
print(f"Device: {device}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"Whisper: ✓ Opérationnel")

if hf_token and hf_token.startswith("hf_"):
    print(f"Pyannote: ✓ Token configuré")
else:
    print(f"Pyannote: ⚠ Token manquant ou invalide")

print("\n✓ Installation validée!")
print("\nProchaines étapes:")
print("1. Configurez votre token HF dans .env (si pas déjà fait)")
print("2. Placez un fichier audio dans data/input/")
print("3. Lancez: python examples/cli_transcribe.py data/input/votre_audio.wav")
print("\nOu utilisez le mode Whisper-only:")
print("python examples/whisper_only.py")
