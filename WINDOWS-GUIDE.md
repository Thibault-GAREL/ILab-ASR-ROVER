# Quick Start Guide - Windows

## Problèmes Résolus

### 1. Erreur TypeError: unexpected keyword argument

**Cause:** `config.yaml` contenait des paramètres non supportés

**Solution:** Utilisez la config mise à jour ou `config-cpu.yaml`

### 2. Warning FFmpeg

**Cause:** FFmpeg n'est pas installé (nécessaire pour certains formats audio)

**Solutions:**

#### Option A: Installer FFmpeg (recommandé)
1. Téléchargez FFmpeg: https://www.gyan.dev/ffmpeg/builds/
2. Extrayez l'archive
3. Ajoutez le dossier `bin` au PATH Windows
4. Vérifiez: `ffmpeg -version`

#### Option B: Utiliser uniquement WAV
- Convertissez vos fichiers en WAV avant transcription
- Pas besoin de FFmpeg pour WAV

### 3. Pas de fichier audio de test

**Solution:** Créez un fichier audio de test

```python
# test_audio.py - Générer un audio de test
import numpy as np
import soundfile as sf

# Générer un signal sinusoïdal simple (bip)
duration = 5  # secondes
sample_rate = 16000
frequency = 440  # Hz (note La)

t = np.linspace(0, duration, int(sample_rate * duration))
audio = 0.3 * np.sin(2 * np.pi * frequency * t)

# Sauvegarder
sf.write("data/input/test_audio.wav", audio, sample_rate)
print("✓ Fichier test créé: data/input/test_audio.wav")
```

## Installation Rapide Windows

```powershell
# 1. Créer environnement virtuel
python -m venv venv
venv\Scripts\activate

# 2. Mettre à jour pip
python -m pip install --upgrade pip

# 3. Installer (Whisper-only)
pip install -r requirements-whisper-only.txt

# 4. Configurer token HuggingFace
copy .env.example .env
# Éditer .env avec Notepad et ajouter votre HF_TOKEN

# 5. Tester avec config CPU
python examples/whisper_only.py
```

## Test Sans Audio Réel

Si vous n'avez pas de fichier audio, utilisez ce script de test :

```python
# test_simple.py
from src.diarization.pyannote_diarizer import PyannoteDiarizer
from src.asr.whisper_asr import WhisperASR

# Test 1: Diarizer
print("Test 1: Initialisation Diarizer...")
diarizer = PyannoteDiarizer(
    model_name="pyannote/speaker-diarization-3.1",
    device="cpu",
    hf_token="votre_token_ici"
)
print("✓ Diarizer OK")

# Test 2: Whisper
print("\nTest 2: Initialisation Whisper...")
whisper = WhisperASR(
    model_size="base",
    device="cpu",
    compute_type="int8"
)
print("✓ Whisper OK")

print("\n✓ Tous les tests passés!")
```

## Configuration CPU (Pas de GPU)

Si vous n'avez pas de GPU CUDA:

```bash
# Utiliser la config CPU
python examples/cli_transcribe.py audio.wav --config configs/config-cpu.yaml
```

Ou en Python:

```python
from src.pipeline import MeetingTranscriptionPipeline

pipeline = MeetingTranscriptionPipeline(
    diarizer_config={
        "model_name": "pyannote/speaker-diarization-3.1",
        "device": "cpu",
        "hf_token": "votre_token"
    },
    whisper_config={
        "model_size": "base",  # Plus petit pour CPU
        "device": "cpu",
        "compute_type": "int8"
    },
    use_canary=False
)
```

## Obtenir un Token HuggingFace

1. Allez sur: https://huggingface.co/settings/tokens
2. Cliquez "New token"
3. Donnez un nom (ex: "pyannote-asr")
4. Sélectionnez "Read"
5. Créez et copiez le token
6. Ajoutez dans `.env`:
   ```
   HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
   ```

## Erreurs Communes

### "No such file: meeting.wav"
```bash
# Créez le dossier et ajoutez un fichier
mkdir data\input
# Copiez votre fichier audio dans data\input\
copy C:\chemin\vers\audio.wav data\input\meeting.wav
```

### "CUDA not available"
**Solution:** Utilisez `config-cpu.yaml` ou changez device="cpu" dans config

### "Invalid HuggingFace token"
**Solution:**
1. Vérifiez que le token commence par `hf_`
2. Acceptez les conditions d'utilisation sur:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0

## Performance CPU vs GPU

| Composant | CPU (base) | GPU (large-v3) |
|-----------|------------|----------------|
| Whisper | ~10x RT | ~70x RT |
| Diarization | ~0.5x RT | ~2.5x RT |
| Total | **Lent** | **Rapide** |

**Recommandation CPU:** Utilisez `model_size: "base"` ou `"small"` pour Whisper

## Exemple Complet Windows

```powershell
# Terminal PowerShell

# 1. Setup
cd C:\0-Code_py_temp\ASR-ROVER
python -m venv venv
venv\Scripts\activate
pip install -r requirements-whisper-only.txt

# 2. Config
copy .env.example .env
notepad .env  # Ajouter HF_TOKEN=hf_...

# 3. Créer audio de test
python -c "import numpy as np, soundfile as sf; sf.write('data/input/test.wav', 0.3*np.sin(2*np.pi*440*np.linspace(0,5,80000)), 16000)"

# 4. Transcrire
python examples/cli_transcribe.py data/input/test.wav --whisper-only --config configs/config-cpu.yaml

# 5. Vérifier résultats
dir data\output
type data\output\test_transcript.txt
```

## Aide Supplémentaire

Si les problèmes persistent:
1. Vérifiez la version Python: `python --version` (doit être 3.8+)
2. Vérifiez pip: `pip --version`
3. Listez les packages: `pip list`
4. Consultez INSTALL.md pour plus de détails
