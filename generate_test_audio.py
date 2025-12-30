"""
Génère un fichier audio de test simple
Utile pour tester le système sans avoir de vrai fichier audio
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Créer le dossier de sortie
output_dir = Path("data/input")
output_dir.mkdir(parents=True, exist_ok=True)

print("Génération d'un fichier audio de test...")

# Paramètres
sample_rate = 16000
duration = 10  # secondes

# Générer un signal simple avec deux "speakers" simulés
t = np.linspace(0, duration, int(sample_rate * duration))

# Speaker 1: Fréquence basse (simule une voix grave)
freq1 = 200  # Hz
speaker1 = 0.3 * np.sin(2 * np.pi * freq1 * t)

# Speaker 2: Fréquence haute (simule une voix aiguë)
freq2 = 400  # Hz
speaker2 = 0.3 * np.sin(2 * np.pi * freq2 * t)

# Alterner entre les deux "speakers"
audio = np.zeros_like(t)
segment_duration = sample_rate * 2  # 2 secondes par segment

for i in range(5):  # 5 segments
    start = i * segment_duration
    end = min((i + 1) * segment_duration, len(audio))

    if i % 2 == 0:
        audio[start:end] = speaker1[start:end]
    else:
        audio[start:end] = speaker2[start:end]

# Sauvegarder
output_path = output_dir / "test_meeting.wav"
sf.write(output_path, audio, sample_rate)

print(f"\n✓ Fichier audio de test créé: {output_path}")
print(f"  Durée: {duration}s")
print(f"  Sample rate: {sample_rate}Hz")
print(f"  Format: WAV mono")
print(f"\nUtilisez-le avec:")
print(f"  python examples/cli_transcribe.py {output_path}")
