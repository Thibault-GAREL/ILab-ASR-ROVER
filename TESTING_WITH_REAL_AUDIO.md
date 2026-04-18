# Test avec un Fichier Audio Réel

## 🎯 Pourquoi les fichiers de test synthétiques ne fonctionnent pas bien

Les fichiers générés par `generate_test_audio.py` sont de simples **tonalités sinusoïdales**. Le modèle Pyannote utilise un détecteur d'activité vocale (VAD) entraîné sur de la vraie parole humaine, qui ne reconnaît pas ces tonalités comme de la parole.

**Résultat** : 0 segments détectés, car le VAD ne détecte aucune "voix" dans le fichier.

## ✅ Solution 1 : Générateur Audio Réaliste

Utilisez le nouveau générateur qui simule de la parole avec des formants vocaux :

```bash
python generate_realistic_test_audio.py
```

Ce générateur crée un fichier avec :
- 2 speakers simulés (voix grave et aiguë)
- Formants vocaux (fréquences de résonance de la voix)
- Harmoniques (comme dans une vraie voix)
- Enveloppes d'attaque et de décroissance

Puis testez :
```bash
# Modifiez test_diarization_fix.py pour utiliser le nouveau fichier
# Ou lancez directement :
python examples/cli_transcribe.py data/input/realistic_test_meeting.wav --whisper-only
```

**Note** : Même avec ce fichier amélioré, la diarisation peut donner des résultats limités car ce n'est toujours pas de vraie parole.

## ✅ Solution 2 : Utiliser un Vrai Fichier Audio (RECOMMANDÉ)

### Télécharger des échantillons audio gratuits

1. **Librivox** (domaine public) :
   - https://librivox.org/
   - Livres audio libres de droits
   - Format MP3/OGG (convertir en WAV)

2. **Common Voice (Mozilla)** :
   - https://commonvoice.mozilla.org/
   - Enregistrements de voix en français et anglais
   - Licence CC0 (domaine public)

3. **VoxPopuli** :
   - Enregistrements du Parlement Européen
   - Multilingue
   - Disponible sur HuggingFace Datasets

### Convertir en WAV si nécessaire

```bash
# Avec FFmpeg (si installé)
ffmpeg -i votre_fichier.mp3 -ar 16000 -ac 1 data/input/meeting.wav

# Ou avec Python
pip install pydub
python -c "from pydub import AudioSegment; AudioSegment.from_mp3('input.mp3').set_channels(1).set_frame_rate(16000).export('data/input/meeting.wav', format='wav')"
```

### Tester avec votre fichier

```bash
# Test de diarisation seule
python -c "
from src.diarization import PyannoteDiarizer
import os

diarizer = PyannoteDiarizer(
    model_name='pyannote/speaker-diarization-3.1',
    device='cpu',
    hf_token=os.getenv('HF_TOKEN')
)

result = diarizer.diarize('data/input/meeting.wav')
print(f'Speakers: {result.num_speakers}')
print(f'Segments: {len(result.segments)}')
for seg in result.segments[:10]:
    print(f'{seg.speaker}: {seg.start:.1f}s - {seg.end:.1f}s')
"

# Test complet avec transcription
python examples/cli_transcribe.py data/input/meeting.wav --whisper-only
```

## 📊 Résultats Attendus avec un Vrai Fichier Audio

### Diarisation seule

```
Speakers: 2-3 (selon votre fichier)
Segments: 50-200 (selon la durée et les pauses)

Exemple de sortie:
SPEAKER_00: 0.0s - 3.5s
SPEAKER_01: 3.8s - 7.2s
SPEAKER_00: 7.5s - 11.3s
...
```

### Transcription complète

```json
{
  "text": "Bonjour à tous, bienvenue à cette réunion...",
  "language": "fr",
  "segments": [
    {
      "speaker": "SPEAKER_00",
      "start": 0.0,
      "end": 3.5,
      "text": "Bonjour à tous, bienvenue à cette réunion."
    },
    {
      "speaker": "SPEAKER_01",
      "start": 3.8,
      "end": 7.2,
      "text": "Merci. Je vais commencer par..."
    }
  ]
}
```

## 🎥 Extraire l'Audio d'une Vidéo

Si vous avez une vidéo de réunion Zoom/Teams :

```bash
# Avec FFmpeg
ffmpeg -i meeting.mp4 -vn -ar 16000 -ac 1 data/input/meeting.wav

# -vn : pas de vidéo
# -ar 16000 : sample rate 16kHz
# -ac 1 : mono
```

## 🎙️ Enregistrer votre Propre Audio de Test

Utilisez Audacity (gratuit) ou tout autre logiciel :

1. Enregistrez 30-60 secondes de conversation (vous + quelqu'un d'autre)
2. Exportez en WAV, 16kHz, mono
3. Sauvegardez dans `data/input/my_test.wav`
4. Testez avec :
   ```bash
   python examples/cli_transcribe.py data/input/my_test.wav --whisper-only --language fr
   ```

## 🔍 Vérifier la Qualité Audio

Avant de tester, vérifiez que votre fichier audio est correct :

```python
import soundfile as sf
import numpy as np

# Lire le fichier
audio, sr = sf.read('data/input/meeting.wav')

print(f"Sample rate: {sr} Hz (optimal: 16000 Hz)")
print(f"Durée: {len(audio)/sr:.2f} secondes")
print(f"Channels: {audio.ndim} (1=mono, 2=stereo)")
print(f"Amplitude max: {np.max(np.abs(audio)):.3f} (optimal: 0.5-1.0)")
print(f"Contient du silence? {np.max(np.abs(audio)) < 0.01}")

# Si stéréo, convertir en mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)
    sf.write('data/input/meeting.wav', audio, sr)
    print("✓ Converti en mono")
```

## ⚙️ Optimiser les Paramètres pour Votre Audio

Si vous avez des problèmes de diarisation :

```python
from src.diarization import PyannoteDiarizer
import os

diarizer = PyannoteDiarizer(
    model_name='pyannote/speaker-diarization-3.1',
    device='cpu',
    hf_token=os.getenv('HF_TOKEN'),
    min_speakers=1,   # Ajustez selon votre cas
    max_speakers=10   # Ajustez selon votre cas
)

# Testez avec différents nombres de speakers
result = diarizer.diarize(
    'data/input/meeting.wav',
    num_speakers=2  # Forcez le nombre si vous le connaissez
)
```

## 📝 Recommandations

1. **Pour tester le système rapidement** :
   - Utilisez `generate_realistic_test_audio.py`
   - Attendez-vous à des résultats limités

2. **Pour tester la vraie performance** :
   - Téléchargez un fichier de Librivox ou Common Voice
   - Ou utilisez une vraie réunion enregistrée

3. **Pour évaluer la qualité** :
   - Utilisez un fichier avec 2-3 speakers clairement identifiables
   - Durée : 1-5 minutes pour commencer
   - Qualité audio : bonne (peu de bruit de fond)

## 🆘 Problèmes Courants

### "0 segments détectés"
- Le fichier est du silence ou du bruit pur
- Utilisez un vrai fichier audio de parole

### "1 seul speaker détecté alors qu'il y en a plusieurs"
- Les voix sont trop similaires
- Ajustez `max_speakers` plus haut
- Essayez avec `num_speakers` forcé

### "Trop de speakers détectés"
- Beaucoup de bruit ou d'échos
- Réduisez `max_speakers`
- Améliorez la qualité audio

## 📚 Ressources

- **Datasets audio gratuits** : https://huggingface.co/datasets?task_categories=task_categories:automatic-speech-recognition
- **Pyannote documentation** : https://github.com/pyannote/pyannote-audio
- **Common Voice** : https://commonvoice.mozilla.org/fr/datasets
- **Librivox** : https://librivox.org/

---

**En résumé** : Pour tester vraiment le système, utilisez un fichier audio avec de **vraie parole humaine**, pas des tonalités synthétiques ! 🎤
