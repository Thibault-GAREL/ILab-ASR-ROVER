"""
Test de diagnostic pour vérifier le fix DiarizeOutput
"""

import sys
from pathlib import Path

# Ajouter le chemin du projet
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TEST DIAGNOSTIC - FIX DIARIZEOUTPUT")
print("=" * 70)

# Charger le token
print("\n[1/4] Chargement du token HuggingFace...")
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    print("❌ Pas de token HF_TOKEN trouvé")
    print("→ Configurez d'abord: python setup_token.py")
    sys.exit(1)

print(f"✓ Token trouvé: {hf_token[:10]}...")

# Vérifier le fichier audio
print("\n[2/4] Vérification du fichier audio...")
audio_file = "data/input/test_meeting.wav"
if not Path(audio_file).exists():
    audio_file = "data/input/meeting.wav"
    if not Path(audio_file).exists():
        print("❌ Pas de fichier audio trouvé")
        print("→ Générez-en un: python generate_test_audio.py")
        sys.exit(1)

print(f"✓ Fichier trouvé: {audio_file}")

# Importer et initialiser le diarizer
print("\n[3/4] Initialisation du diarizer...")
try:
    from src.diarization.pyannote_diarizer import PyannoteDiarizer

    diarizer = PyannoteDiarizer(
        model_name="pyannote/speaker-diarization-3.1",
        device="cpu",
        hf_token=hf_token
    )
    print("✓ Diarizer initialisé")
except Exception as e:
    print(f"❌ Erreur initialisation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test de diarisation
print("\n[4/4] Test de diarisation...")
print("(Cela peut prendre quelques minutes...)")
try:
    result = diarizer.diarize(audio_file)

    print("\n" + "=" * 70)
    print("✓ SUCCÈS! Le fix fonctionne!")
    print("=" * 70)
    print(f"\nNombre de segments: {len(result.segments)}")
    print(f"Speakers détectés: {len(result.speakers)}")
    print(f"\nPremiers segments:")
    for i, seg in enumerate(result.segments[:5]):
        print(f"  {seg.start:.1f}s - {seg.end:.1f}s: {seg.speaker}")

except AttributeError as e:
    print("\n" + "=" * 70)
    print("❌ ERREUR: Le fix ne fonctionne pas encore")
    print("=" * 70)
    print(f"Erreur: {e}")
    import traceback
    traceback.print_exc()

    print("\n→ Vérifiez que vous avez bien:")
    print("   1. Vidé le cache: find . -type d -name __pycache__ -exec rm -rf {} +")
    print("   2. Redémarré votre terminal/IDE")
    print("   3. Tiré les dernières modifications: git pull")

except Exception as e:
    print(f"\n❌ Autre erreur: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("FIN DU TEST DIAGNOSTIC")
print("=" * 70)
