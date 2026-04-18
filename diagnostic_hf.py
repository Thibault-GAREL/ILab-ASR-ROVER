"""
Diagnostic des accès HuggingFace
Identifie exactement quel modèle pose problème
"""

import os
import sys

# Charger le token
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

token = os.getenv("HF_TOKEN")

if not token:
    print("❌ Aucun token HF_TOKEN trouvé")
    sys.exit(1)

print("=" * 70)
print("DIAGNOSTIC DES ACCÈS HUGGINGFACE")
print("=" * 70)
print(f"\n✓ Token trouvé: {token[:10]}...")

# Se connecter à HuggingFace
print("\n[1/5] Connexion à HuggingFace...")
try:
    from huggingface_hub import login, whoami
    login(token=token, add_to_git_credential=False)
    user_info = whoami(token=token)
    print(f"✓ Connecté en tant que: {user_info['name']}")
except Exception as e:
    print(f"❌ Erreur de connexion: {e}")
    sys.exit(1)

# Liste des modèles requis
models_to_check = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "pyannote/wespeaker-voxceleb-resnet34-LM",
    "pyannote/embedding",
]

print("\n[2/5] Vérification des accès aux modèles...")
from huggingface_hub import model_info

for model_name in models_to_check:
    try:
        info = model_info(model_name, token=token)
        print(f"✓ {model_name}")
        if hasattr(info, 'gated') and info.gated:
            print(f"  ℹ Modèle avec accès restreint (gated)")
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg or "gated" in error_msg.lower():
            print(f"❌ {model_name}")
            print(f"  → Accès refusé - Vous devez accepter la licence!")
            print(f"  → Allez sur: https://huggingface.co/{model_name}")
            print(f"  → Cliquez sur 'Agree and access repository'")
        else:
            print(f"⚠ {model_name}: {error_msg}")

# Tester le téléchargement du pipeline
print("\n[3/5] Test de téléchargement du pipeline...")
try:
    from pyannote.audio import Pipeline
    print("Tentative de chargement de pyannote/speaker-diarization-3.1...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=token
    )
    print("✓ Pipeline chargé avec succès!")
    print("\n🎉 TOUT FONCTIONNE! Vous pouvez maintenant utiliser le système.")
except Exception as e:
    print(f"❌ Échec du chargement: {e}")
    print("\n📋 ACTIONS REQUISES:")

    # Identifier quel modèle pose problème
    error_msg = str(e)
    if "segmentation" in error_msg:
        print("\n→ Problème avec le modèle de segmentation")
        print("   Allez sur: https://huggingface.co/pyannote/segmentation-3.0")
        print("   Cliquez 'Agree and access repository'")
    elif "embedding" in error_msg or "wespeaker" in error_msg:
        print("\n→ Problème avec le modèle d'embedding")
        print("   Allez sur: https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM")
        print("   Cliquez 'Agree and access repository'")
    else:
        print("\n→ Vérifiez que vous avez accepté TOUTES les licences:")
        for model in models_to_check:
            print(f"   - https://huggingface.co/{model}")

print("\n[4/5] Vérification du cache HuggingFace...")
cache_dir = os.path.expanduser("~/.cache/huggingface")
if os.path.exists(cache_dir):
    print(f"✓ Cache trouvé: {cache_dir}")
else:
    print(f"⚠ Pas de cache HuggingFace (normal si première utilisation)")

print("\n[5/5] Test de connexion sans cache...")
print("Si vous avez des problèmes, essayez:")
print("  1. Attendez 5-10 minutes (délai de propagation HuggingFace)")
print("  2. Videz le cache: rmdir /s %USERPROFILE%\\.cache\\huggingface")
print("  3. Reconnectez-vous: huggingface-cli login")

print("\n" + "=" * 70)
print("FIN DU DIAGNOSTIC")
print("=" * 70)
