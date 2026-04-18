"""
Guide de Configuration Complète - Token HuggingFace
Suivez ces étapes DANS L'ORDRE pour que tout fonctionne
"""

import os
import sys

print("=" * 70)
print("GUIDE DE CONFIGURATION - ASR ROVER")
print("=" * 70)

# Étape 1: Vérifier si .env existe
print("\n[ÉTAPE 1] Vérification du fichier .env")
if os.path.exists(".env"):
    print("✓ Fichier .env trouvé")
    with open(".env", "r") as f:
        content = f.read()
        if "HF_TOKEN=" in content and "your_huggingface_token_here" not in content:
            print("✓ HF_TOKEN configuré dans .env")
            # Extraire le token
            for line in content.split("\n"):
                if line.startswith("HF_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    if token.startswith("hf_"):
                        print(f"✓ Token valide trouvé: {token[:10]}...")
                    else:
                        print(f"✗ Token invalide (doit commencer par 'hf_'): {token[:10]}")
        else:
            print("✗ HF_TOKEN non configuré ou valeur par défaut")
            print("\n→ ACTION REQUISE:")
            print("   1. Allez sur: https://huggingface.co/settings/tokens")
            print("   2. Créez un nouveau token (type: Read)")
            print("   3. Copiez le token (commence par hf_)")
            print("   4. Éditez .env avec: notepad .env")
            print("   5. Remplacez 'your_huggingface_token_here' par votre token")
else:
    print("✗ Fichier .env introuvable")
    print("\n→ ACTION REQUISE:")
    print("   Exécutez: copy .env.example .env")
    print("   Puis éditez avec: notepad .env")

# Étape 2: Tester le chargement du token
print("\n[ÉTAPE 2] Test de chargement du token")
try:
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("HF_TOKEN")

    if token:
        if token.startswith("hf_") and len(token) > 20:
            print(f"✓ Token chargé avec succès: {token[:10]}...")
        else:
            print(f"✗ Token invalide: {token[:20]}")
    else:
        print("✗ HF_TOKEN non trouvé dans les variables d'environnement")
        print("\n→ Le fichier .env n'est pas chargé correctement")
except ImportError:
    print("✗ python-dotenv non installé")
    print("→ Exécutez: pip install python-dotenv")

# Étape 3: Vérifier l'accès aux modèles
print("\n[ÉTAPE 3] Vérification de l'accès aux modèles Pyannote")
print("\nVous DEVEZ accepter les licences sur ces pages:")
print("→ https://huggingface.co/pyannote/speaker-diarization-3.1")
print("→ https://huggingface.co/pyannote/segmentation-3.0")
print("\nCliquez sur 'Agree and access repository' sur chaque page")

# Étape 4: Test complet
print("\n[ÉTAPE 4] Test de connexion à HuggingFace")
if 'token' in locals() and token and token.startswith("hf_"):
    print("Tentative de connexion...")
    try:
        from huggingface_hub import login
        login(token=token)
        print("✓ Connexion réussie à HuggingFace!")
    except Exception as e:
        print(f"✗ Erreur de connexion: {e}")
        print("\n→ Vérifiez que:")
        print("   1. Le token est valide")
        print("   2. Vous avez accepté les licences des modèles")
else:
    print("⊘ Test ignoré (pas de token valide)")

# Résumé
print("\n" + "=" * 70)
print("RÉSUMÉ ET PROCHAINES ÉTAPES")
print("=" * 70)
print("\n1. Si tout est ✓ ci-dessus:")
print("   python examples/cli_transcribe.py data/input/meeting.wav --whisper-only")
print("\n2. Si vous avez des ✗:")
print("   - Suivez les ACTIONS REQUISES mentionnées")
print("   - Relancez ce script: python setup_token.py")
print("\n3. Pour tester sans diarisation:")
print("   python test_whisper_seul.py")
