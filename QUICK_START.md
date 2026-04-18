# 🚀 Guide de Démarrage Rapide

Guide complet pour utiliser le système de transcription multilingue avec diarisation et ROVER.

## 📋 Prérequis

- Python 3.8 ou supérieur
- ~5GB d'espace disque (pour les modèles)
- Connexion internet (pour télécharger les modèles)
- Token HuggingFace (gratuit)

## 🔧 Installation Complète

### Option A : Script automatique (Recommandé)

```bash
# 1. Installer toutes les dépendances
bash install_dependencies.sh

# 2. Configurer le token HuggingFace
python setup_token.py

# 3. Générer un fichier audio de test
python generate_test_audio.py

# 4. Tester le système
python test_diarization_fix.py
```

### Option B : Installation manuelle

```bash
# 1. Installer les dépendances
pip install -r requirements-whisper-only.txt

# 2. Vider le cache Python (important!)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# 3. Configurer le token HuggingFace
python setup_token.py

# 4. Tester
python generate_test_audio.py
python test_diarization_fix.py
```

## 🎯 Utilisation

### 1. Transcription Simple (Whisper uniquement)

```bash
python examples/cli_transcribe.py votre_fichier.wav --whisper-only
```

**Ce mode utilise :**
- ✅ Whisper Large V3 pour la transcription
- ✅ Pyannote pour la diarisation (qui a parlé quand)
- ✅ Supporte 99 langues (dont français et anglais)
- ⚡ Pas de ROVER (plus rapide, un seul modèle)

**Exemple de sortie :**
```json
{
  "text": "Bonjour à tous. Bienvenue à cette réunion.",
  "language": "fr",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 2.5,
      "text": "Bonjour à tous."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 2.5,
      "end": 5.0,
      "text": "Bienvenue à cette réunion."
    }
  ]
}
```

### 2. Transcription avec ROVER (Maximum de précision)

```bash
python examples/cli_transcribe.py votre_fichier.wav
```

**Ce mode utilise :**
- ✅ Whisper Large V3 (robustesse)
- ✅ NVIDIA Canary (précision)
- ✅ ROVER pour fusionner les résultats
- ✅ Diarisation Pyannote
- ⚠️ Nécessite `requirements-full.txt` (avec NeMo)

**Note :** L'installation complète avec NeMo peut avoir des conflits de dépendances. Utilisez `--whisper-only` si vous rencontrez des problèmes.

### 3. Utilisation Programmatique (Python)

```python
from src.pipeline import TranscriptionPipeline

# Initialiser le pipeline
pipeline = TranscriptionPipeline(
    config_path="configs/config.yaml",
    whisper_only=True  # ou False pour utiliser ROVER
)

# Transcrire un fichier
result = pipeline.transcribe("meeting.wav")

# Accéder aux résultats
print(f"Transcription: {result.text}")
print(f"Langue détectée: {result.language}")
print(f"Nombre de speakers: {len(result.speakers)}")

# Afficher les segments avec speakers
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s] "
          f"{segment.speaker}: {segment.text}")
```

## 🎛️ Configuration

### Fichier de configuration : `configs/config.yaml`

```yaml
# Diarisation (qui a parlé quand)
diarization:
  model_name: "pyannote/speaker-diarization-3.1"
  device: "cpu"  # ou "cuda" si GPU disponible
  min_speakers: 1
  max_speakers: 10

# Modèle Whisper
whisper:
  model_size: "large-v3"
  device: "cpu"
  compute_type: "int8"  # "float16" sur GPU
  language: null  # null = détection auto, ou "fr", "en"
  beam_size: 5
  best_of: 5

# Fusion ROVER (si --whisper-only n'est pas utilisé)
rover:
  confidence_weight: 0.6
  model_weight: 0.4
```

### Variables d'environnement : `.env`

```bash
# Token HuggingFace (obligatoire pour Pyannote)
HF_TOKEN=hf_votre_token_ici

# Optionnel : activer les logs détaillés
LOG_LEVEL=INFO
```

## 🔍 Dépannage

### Erreur : "DiarizeOutput has no attribute 'itertracks'"

**Solution :** Problème de cache Python
```bash
# Vider le cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find . -type f -name "*.pyc" -delete 2>/dev/null

# Redémarrer votre terminal
# Puis relancer
```

Voir le guide complet : `FIX_CACHE_PYTHON.md`

### Erreur : "401 Unauthorized" pour Pyannote

**Solution :** Accepter les licences des modèles

1. Allez sur HuggingFace et acceptez les conditions pour :
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1
   - https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
   - https://huggingface.co/pyannote/VoiceActivityDetection-PyanNet-ONNX

2. Lancez le diagnostic :
```bash
python diagnostic_hf.py
```

### Erreur : Conflits de dépendances avec NeMo

**Solution :** Utilisez le mode Whisper-only
```bash
# Désinstallez NeMo si installé
pip uninstall nemo-toolkit -y

# Réinstallez les dépendances simples
pip install -r requirements-whisper-only.txt

# Utilisez --whisper-only
python examples/cli_transcribe.py audio.wav --whisper-only
```

### Performance lente

**Solutions :**

1. **Utiliser un GPU :**
   ```yaml
   # Dans config.yaml
   diarization:
     device: "cuda"
   whisper:
     device: "cuda"
     compute_type: "float16"
   ```

2. **Utiliser un modèle Whisper plus petit :**
   ```yaml
   whisper:
     model_size: "medium"  # au lieu de "large-v3"
   ```

3. **Optimiser les paramètres :**
   ```yaml
   whisper:
     beam_size: 1  # au lieu de 5 (plus rapide, légèrement moins précis)
     best_of: 1    # au lieu de 5
   ```

## 📊 Benchmarks et Performances

### Word Error Rate (WER) attendu

| Langue    | Modèle        | WER    |
|-----------|---------------|--------|
| Anglais   | Whisper Large | ~7-8%  |
| Français  | Whisper Large | ~8-10% |
| Anglais   | ROVER         | ~5-6%  |

### Temps de traitement (CPU)

| Durée Audio | Temps de Traitement | RTFx  |
|-------------|---------------------|-------|
| 1 minute    | ~2-3 minutes        | 2-3x  |
| 10 minutes  | ~20-30 minutes      | 2-3x  |
| 1 heure     | ~2-3 heures         | 2-3x  |

**RTFx (Real-Time Factor)** : 2-3x signifie que le traitement prend 2-3 fois la durée de l'audio.

### Utilisation GPU

Avec un GPU NVIDIA (CUDA), les performances sont bien meilleures :

| Durée Audio | Temps (GPU) | RTFx  |
|-------------|-------------|-------|
| 1 minute    | ~10-20s     | 0.2x  |
| 10 minutes  | ~2-3 min    | 0.2x  |
| 1 heure     | ~12-18 min  | 0.2x  |

## 🎓 Exemples Complets

### Exemple 1 : Réunion bilingue

```bash
# Fichier : meeting_fr_en.wav
python examples/cli_transcribe.py meeting_fr_en.wav \
  --whisper-only \
  --output meeting_fr_en.json

# Résultat dans meeting_fr_en.json avec détection automatique de langue
```

### Exemple 2 : Forcer la langue française

```python
from src.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline(
    config_path="configs/config.yaml",
    whisper_only=True
)

# Modifier la config pour forcer le français
pipeline.config['whisper']['language'] = 'fr'

result = pipeline.transcribe("reunion.wav")
```

### Exemple 3 : Traiter plusieurs fichiers

```python
import glob
from src.pipeline import TranscriptionPipeline

pipeline = TranscriptionPipeline(whisper_only=True)

# Traiter tous les WAV dans un dossier
for audio_file in glob.glob("data/input/*.wav"):
    print(f"Traitement de {audio_file}...")
    result = pipeline.transcribe(audio_file)

    # Sauvegarder les résultats
    output_file = audio_file.replace('.wav', '_transcript.json')
    import json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'text': result.text,
            'language': result.language,
            'segments': [
                {
                    'speaker': s.speaker,
                    'start': s.start,
                    'end': s.end,
                    'text': s.text
                }
                for s in result.segments
            ]
        }, f, ensure_ascii=False, indent=2)

    print(f"Sauvegardé dans {output_file}")
```

## 🔗 Ressources

- **Documentation Whisper :** https://github.com/openai/whisper
- **Documentation Pyannote :** https://github.com/pyannote/pyannote-audio
- **ROVER Paper :** https://www.nist.gov/publications/rover-recognizer-output-voting-error-reduction
- **HuggingFace :** https://huggingface.co/

## 💡 Conseils d'utilisation

1. **Pour les réunions longues (>1h) :**
   - Utilisez `--whisper-only` (plus stable)
   - Activez le GPU si possible
   - Divisez en segments de ~30 minutes si problèmes de mémoire

2. **Pour la meilleure précision :**
   - Utilisez ROVER (mode complet, pas --whisper-only)
   - Audio de bonne qualité (peu de bruit de fond)
   - Spécifiez la langue si connue

3. **Pour le développement/tests :**
   - Utilisez des modèles plus petits : `medium` ou `small`
   - Générez des fichiers audio de test courts
   - Activez les logs : `LOG_LEVEL=DEBUG`

## ❓ Support

Pour des problèmes ou questions :

1. Consultez `FIX_CACHE_PYTHON.md` pour les problèmes de cache
2. Consultez `WINDOWS-GUIDE.md` pour les problèmes Windows
3. Lancez `python diagnostic_hf.py` pour les problèmes HuggingFace
4. Ouvrez une issue sur GitHub avec les logs complets

---

**Bon usage de votre système de transcription multilingue ! 🎉**
