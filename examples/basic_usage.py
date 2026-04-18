"""
Exemples d'utilisation basique du système de transcription multilingue
"""

import sys
from pathlib import Path

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TranscriptionPipeline
import json


def exemple_1_transcription_simple():
    """
    Exemple 1 : Transcription simple d'un fichier audio
    """
    print("=" * 70)
    print("EXEMPLE 1 : Transcription Simple")
    print("=" * 70)

    # Initialiser le pipeline (mode Whisper-only pour éviter les dépendances NeMo)
    pipeline = TranscriptionPipeline(
        config_path="configs/config.yaml",
        whisper_only=True
    )

    # Transcrire un fichier
    audio_file = "data/input/test_meeting.wav"
    print(f"\nTranscription de : {audio_file}")
    print("Cela peut prendre quelques minutes...\n")

    result = pipeline.transcribe(audio_file)

    # Afficher les résultats
    print(f"Transcription complète :")
    print(f"  {result.text}\n")
    print(f"Langue détectée : {result.language}")
    print(f"Nombre de speakers : {len(result.speakers)}")
    print(f"Nombre de segments : {len(result.segments)}\n")


def exemple_2_avec_speakers():
    """
    Exemple 2 : Transcription avec identification des speakers
    """
    print("=" * 70)
    print("EXEMPLE 2 : Transcription avec Speakers")
    print("=" * 70)

    pipeline = TranscriptionPipeline(
        config_path="configs/config.yaml",
        whisper_only=True
    )

    audio_file = "data/input/test_meeting.wav"
    result = pipeline.transcribe(audio_file)

    # Afficher chaque segment avec son speaker
    print("\nSegments avec speakers :")
    print("-" * 70)
    for i, segment in enumerate(result.segments, 1):
        print(f"[{i}] {segment.speaker} "
              f"({segment.start:.1f}s - {segment.end:.1f}s)")
        print(f"    {segment.text}\n")


def exemple_3_sauvegarder_json():
    """
    Exemple 3 : Sauvegarder les résultats en JSON
    """
    print("=" * 70)
    print("EXEMPLE 3 : Sauvegarder en JSON")
    print("=" * 70)

    pipeline = TranscriptionPipeline(
        config_path="configs/config.yaml",
        whisper_only=True
    )

    audio_file = "data/input/test_meeting.wav"
    result = pipeline.transcribe(audio_file)

    # Préparer les données pour JSON
    output_data = {
        "text": result.text,
        "language": result.language,
        "speakers": list(result.speakers),
        "segments": [
            {
                "speaker": seg.speaker,
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence
            }
            for seg in result.segments
        ],
        "metadata": {
            "audio_file": audio_file,
            "num_speakers": len(result.speakers),
            "num_segments": len(result.segments)
        }
    }

    # Sauvegarder
    output_file = "data/output/transcript.json"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\nRésultats sauvegardés dans : {output_file}")
    print(f"Taille du fichier : {Path(output_file).stat().st_size} bytes")


def exemple_4_forcer_langue():
    """
    Exemple 4 : Forcer la langue de transcription
    """
    print("=" * 70)
    print("EXEMPLE 4 : Forcer la langue (français)")
    print("=" * 70)

    pipeline = TranscriptionPipeline(
        config_path="configs/config.yaml",
        whisper_only=True
    )

    # Forcer la langue française
    pipeline.config['whisper']['language'] = 'fr'

    audio_file = "data/input/test_meeting.wav"
    result = pipeline.transcribe(audio_file)

    print(f"\nTranscription en français forcé :")
    print(f"  {result.text}")
    print(f"\nLangue détectée : {result.language}")


def exemple_5_traiter_multiple_fichiers():
    """
    Exemple 5 : Traiter plusieurs fichiers en batch
    """
    print("=" * 70)
    print("EXEMPLE 5 : Traitement par lots")
    print("=" * 70)

    pipeline = TranscriptionPipeline(
        config_path="configs/config.yaml",
        whisper_only=True
    )

    # Liste de fichiers à traiter
    audio_files = list(Path("data/input").glob("*.wav"))

    if not audio_files:
        print("\nAucun fichier WAV trouvé dans data/input/")
        return

    print(f"\nTraitement de {len(audio_files)} fichier(s)...\n")

    results = []
    for audio_file in audio_files:
        print(f"Traitement : {audio_file.name}...")

        try:
            result = pipeline.transcribe(str(audio_file))
            results.append({
                "file": audio_file.name,
                "text": result.text,
                "language": result.language,
                "speakers": len(result.speakers)
            })
            print(f"  ✓ Terminé - {len(result.speakers)} speaker(s)")

        except Exception as e:
            print(f"  ✗ Erreur : {e}")

    # Sauvegarder le résumé
    summary_file = "data/output/batch_summary.json"
    Path(summary_file).parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nRésumé sauvegardé dans : {summary_file}")


def exemple_6_configuration_personnalisee():
    """
    Exemple 6 : Utiliser une configuration personnalisée
    """
    print("=" * 70)
    print("EXEMPLE 6 : Configuration Personnalisée")
    print("=" * 70)

    # Configuration personnalisée
    custom_config = {
        'diarization': {
            'model_name': 'pyannote/speaker-diarization-3.1',
            'device': 'cpu',
            'min_speakers': 2,  # Forcer au moins 2 speakers
            'max_speakers': 5   # Maximum 5 speakers
        },
        'whisper': {
            'model_size': 'medium',  # Modèle plus petit = plus rapide
            'device': 'cpu',
            'compute_type': 'int8',
            'language': None,
            'beam_size': 1,  # Plus rapide, légèrement moins précis
            'best_of': 1
        }
    }

    # Initialiser avec la config personnalisée
    pipeline = TranscriptionPipeline(
        config=custom_config,
        whisper_only=True
    )

    audio_file = "data/input/test_meeting.wav"
    result = pipeline.transcribe(audio_file)

    print(f"\nTranscription avec config personnalisée :")
    print(f"  Modèle : medium (plus rapide)")
    print(f"  Speakers : {len(result.speakers)}")
    print(f"  Texte : {result.text[:100]}...")


def menu():
    """
    Menu interactif pour choisir un exemple
    """
    print("\n" + "=" * 70)
    print("EXEMPLES D'UTILISATION - Système de Transcription Multilingue")
    print("=" * 70)
    print("\nChoisissez un exemple :")
    print("  1. Transcription simple")
    print("  2. Transcription avec identification des speakers")
    print("  3. Sauvegarder les résultats en JSON")
    print("  4. Forcer la langue de transcription")
    print("  5. Traiter plusieurs fichiers en batch")
    print("  6. Configuration personnalisée")
    print("  0. Quitter")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    import os

    # Vérifier que le token HuggingFace est configuré
    if not os.getenv("HF_TOKEN"):
        print("⚠️  ATTENTION : Token HuggingFace non configuré")
        print("   Exécutez d'abord : python setup_token.py")
        print("")

    # Vérifier qu'un fichier audio existe
    if not Path("data/input/test_meeting.wav").exists():
        print("⚠️  ATTENTION : Fichier audio de test non trouvé")
        print("   Exécutez d'abord : python generate_test_audio.py")
        print("")

    # Menu interactif
    while True:
        menu()
        try:
            choice = input("Votre choix : ").strip()

            if choice == "0":
                print("\nAu revoir!")
                break
            elif choice == "1":
                exemple_1_transcription_simple()
            elif choice == "2":
                exemple_2_avec_speakers()
            elif choice == "3":
                exemple_3_sauvegarder_json()
            elif choice == "4":
                exemple_4_forcer_langue()
            elif choice == "5":
                exemple_5_traiter_multiple_fichiers()
            elif choice == "6":
                exemple_6_configuration_personnalisee()
            else:
                print("\n✗ Choix invalide")

            input("\n[Appuyez sur Entrée pour continuer]")

        except KeyboardInterrupt:
            print("\n\nInterruption - Au revoir!")
            break
        except Exception as e:
            print(f"\n✗ Erreur : {e}")
            import traceback
            traceback.print_exc()
            input("\n[Appuyez sur Entrée pour continuer]")
