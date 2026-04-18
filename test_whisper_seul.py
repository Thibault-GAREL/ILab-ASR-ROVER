"""
Test Whisper SEUL - Sans diarisation
Utilisez ceci pour tester rapidement sans token HuggingFace
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TEST WHISPER SEUL (SANS DIARISATION)")
print("=" * 70)

# Importer Whisper
from src.asr.whisper_asr import WhisperASR

# Initialiser
print("\n[1/3] Initialisation de Whisper...")
whisper = WhisperASR(
    model_size="base",  # Petit modèle pour CPU
    device="cpu",
    compute_type="int8"
)
print("✓ Whisper initialisé")

# Vérifier le fichier audio
audio_file = "data/input/meeting.wav"
print(f"\n[2/3] Vérification du fichier audio: {audio_file}")

if not Path(audio_file).exists():
    print(f"✗ Fichier introuvable: {audio_file}")
    print("\n→ OPTIONS:")
    print("   1. Générez un fichier de test: python generate_test_audio.py")
    print("   2. Copiez votre fichier dans: data/input/meeting.wav")
    sys.exit(1)

print(f"✓ Fichier trouvé: {audio_file}")

# Transcrire
print(f"\n[3/3] Transcription en cours...")
print("(Cela peut prendre quelques minutes sur CPU...)")

result = whisper.transcribe(audio_file)

# Afficher les résultats
print("\n" + "=" * 70)
print("RÉSULTATS")
print("=" * 70)
print(f"\nLangue détectée: {result.language}")
print(f"Confiance moyenne: {result.confidence:.2%}")
print(f"Nombre de mots: {len(result.words)}")

print("\n--- Transcription ---")
print(result.text)
print("--- Fin ---")

# Sauvegarder
output_file = "transcription_whisper_only.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write(f"Langue: {result.language}\n")
    f.write(f"Confiance: {result.confidence:.2%}\n")
    f.write(f"\n{result.text}\n")

print(f"\n✓ Sauvegardé dans: {output_file}")

print("\n" + "=" * 70)
print("✓ TEST RÉUSSI!")
print("=" * 70)
print("\nPour ajouter la diarisation (identification des speakers):")
print("1. Configurez votre token HF: python setup_token.py")
print("2. Utilisez le pipeline complet: python examples/cli_transcribe.py")
