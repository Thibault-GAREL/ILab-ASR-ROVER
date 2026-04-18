"""
Génère un fichier audio de test plus réaliste qui simule de la parole
avec des formants et du bruit vocal-like
"""

import numpy as np
import soundfile as sf
from pathlib import Path

# Créer le dossier de sortie
output_dir = Path("data/input")
output_dir.mkdir(parents=True, exist_ok=True)

print("Génération d'un fichier audio de test réaliste...")

# Paramètres
sample_rate = 16000
duration = 10  # secondes

def generate_voiced_segment(duration_sec, sample_rate, f0=150, formants=[800, 1200, 2500]):
    """
    Génère un segment qui ressemble plus à de la parole avec formants

    Args:
        duration_sec: Durée en secondes
        sample_rate: Fréquence d'échantillonnage
        f0: Fréquence fondamentale (pitch)
        formants: Fréquences des formants
    """
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0, duration_sec, n_samples)

    # Signal de base (glottal pulse simulation)
    signal = np.zeros(n_samples)

    # Ajouter la fréquence fondamentale et harmoniques
    for harmonic in range(1, 8):
        amplitude = 1.0 / harmonic  # Décroissance des harmoniques
        signal += amplitude * np.sin(2 * np.pi * f0 * harmonic * t)

    # Ajouter des formants (résonances)
    for formant_freq in formants:
        # Modulation pour simuler les formants
        modulation = 0.3 * np.sin(2 * np.pi * formant_freq * t)
        signal += modulation

    # Ajouter du bruit (aspiration, fricatives)
    noise = 0.05 * np.random.randn(n_samples)
    signal += noise

    # Enveloppe (attack, decay)
    envelope = np.ones(n_samples)
    attack_samples = int(0.05 * sample_rate)  # 50ms attack
    decay_samples = int(0.05 * sample_rate)   # 50ms decay

    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        envelope[-decay_samples:] = np.linspace(1, 0, decay_samples)

    signal *= envelope

    # Normaliser
    signal = signal / np.max(np.abs(signal)) * 0.7

    return signal

# Créer un fichier avec deux "speakers" simulés
audio = np.zeros(int(sample_rate * duration))

# Speaker 1: Voix grave (homme)
print("  Génération Speaker 1 (voix grave)...")
segment1 = generate_voiced_segment(2.0, sample_rate, f0=120, formants=[700, 1100, 2500])
audio[0:len(segment1)] = segment1

# Silence
silence_duration = int(sample_rate * 0.5)

# Speaker 2: Voix aiguë (femme)
print("  Génération Speaker 2 (voix aiguë)...")
start_pos = len(segment1) + silence_duration
segment2 = generate_voiced_segment(2.0, sample_rate, f0=220, formants=[850, 1400, 2800])
audio[start_pos:start_pos + len(segment2)] = segment2

# Silence
start_pos = start_pos + len(segment2) + silence_duration

# Speaker 1 à nouveau
print("  Génération Speaker 1 (suite)...")
segment3 = generate_voiced_segment(2.0, sample_rate, f0=120, formants=[700, 1100, 2500])
audio[start_pos:start_pos + len(segment3)] = segment3

# Silence
start_pos = start_pos + len(segment3) + silence_duration

# Speaker 2 à nouveau
print("  Génération Speaker 2 (suite)...")
segment4 = generate_voiced_segment(1.5, sample_rate, f0=220, formants=[850, 1400, 2800])
audio[start_pos:start_pos + len(segment4)] = segment4

# Sauvegarder
output_path = output_dir / "realistic_test_meeting.wav"
sf.write(output_path, audio, sample_rate)

print(f"\n✓ Fichier audio de test réaliste créé: {output_path}")
print(f"  Durée: {duration}s")
print(f"  Sample rate: {sample_rate}Hz")
print(f"  Format: WAV mono")
print(f"  Contenu: 2 speakers simulés avec formants vocaux")
print(f"\nUtilisez-le avec:")
print(f"  python test_diarization_fix.py")
print(f"  (ou modifiez le test pour utiliser 'realistic_test_meeting.wav')")
