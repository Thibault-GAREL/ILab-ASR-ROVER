# FIX : Erreur "DiarizeOutput object has no attribute 'itertracks'"

## 🔍 Problème

Vous obtenez l'erreur suivante même après avoir mis à jour le code :
```
AttributeError: 'DiarizeOutput' object has no attribute 'itertracks'
```

## ✅ Cause

Python utilise des fichiers de cache (`.pyc`) pour accélérer l'import des modules. Même si vous avez mis à jour le fichier source `.py`, Python peut continuer à utiliser l'ancienne version en cache.

## 🛠️ Solution Complète

### Étape 1 : Vérifier que vous avez le dernier code

```bash
# Assurez-vous d'être sur la bonne branche
git status

# Tirez les dernières modifications
git pull origin claude/multilingual-meeting-transcription-BuLON
```

### Étape 2 : Vider TOUS les caches Python

#### Option A : Avec find (Linux/Mac)
```bash
# Supprimer tous les dossiers __pycache__
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Supprimer tous les fichiers .pyc
find . -type f -name "*.pyc" -delete 2>/dev/null
```

#### Option B : Avec PowerShell (Windows)
```powershell
# Supprimer tous les dossiers __pycache__
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force

# Supprimer tous les fichiers .pyc
Get-ChildItem -Path . -Recurse -File -Filter "*.pyc" | Remove-Item -Force
```

#### Option C : Manuel
Supprimez manuellement les dossiers suivants s'ils existent :
- `src/__pycache__/`
- `src/diarization/__pycache__/`
- `src/asr/__pycache__/`
- `src/fusion/__pycache__/`

### Étape 3 : Redémarrer votre terminal/IDE

**IMPORTANT** : Fermez complètement votre terminal ou IDE et rouvrez-le. Cela garantit qu'aucun module Python n'est encore chargé en mémoire.

### Étape 4 : Vérifier le fix manuellement

Ouvrez le fichier `src/diarization/pyannote_diarizer.py` et vérifiez que les lignes 151-156 contiennent bien ce code :

```python
# Extract Annotation from DiarizeOutput if needed
# When passing dict, pyannote returns DiarizeOutput with .diarization attribute
if hasattr(diarization_output, 'diarization'):
    diarization = diarization_output.diarization
else:
    diarization = diarization_output
```

Si ce n'est pas le cas, vous n'avez pas la bonne version du code. Retournez à l'étape 1.

### Étape 5 : Tester avec le script diagnostic

```bash
# Assurez-vous d'avoir un fichier audio
python generate_test_audio.py

# Lancez le test diagnostic
python test_diarization_fix.py
```

## 🎯 Résultats attendus

Si tout fonctionne, vous devriez voir :

```
======================================================================
✓ SUCCÈS! Le fix fonctionne!
======================================================================

Nombre de segments: X
Speakers détectés: Y

Premiers segments:
  0.0s - 2.5s: SPEAKER_00
  2.5s - 5.0s: SPEAKER_01
  ...
```

## ❌ Si ça ne fonctionne toujours pas

### Vérification 1 : Version de Python
```bash
python --version
# Doit être Python 3.8 ou supérieur
```

### Vérification 2 : Dépendances installées
```bash
pip list | grep -E "(pyannote|torch)"

# Vous devriez voir :
# pyannote.audio    X.X.X
# torch             X.X.X
# torchaudio        X.X.X
```

Si pyannote.audio n'est pas installé :
```bash
pip install -r requirements-whisper-only.txt
```

### Vérification 3 : Import manuel pour débugger

Créez un fichier `test_import.py` :

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Afficher le chemin du module
from src.diarization import pyannote_diarizer
print(f"Module chargé depuis : {pyannote_diarizer.__file__}")

# Lire les lignes concernées
with open(pyannote_diarizer.__file__, 'r') as f:
    lines = f.readlines()
    print("\nLignes 151-156:")
    for i in range(150, 156):
        print(f"{i+1}: {lines[i]}", end='')
```

Lancez-le :
```bash
python test_import.py
```

Cela vous montrera exactement quel fichier Python charge et son contenu.

### Vérification 4 : Réinstallation complète

En dernier recours, réinstallez tout :

```bash
# Supprimer l'environnement virtuel si vous en utilisez un
# Ou désinstallez les packages
pip uninstall pyannote.audio torch torchaudio -y

# Vider le cache pip
pip cache purge

# Réinstaller
pip install -r requirements-whisper-only.txt
```

## 📝 Notes Techniques

### Pourquoi ce bug ?

L'API Pyannote a changé récemment. Quand on passe un dictionnaire `{"waveform": ..., "sample_rate": ...}` au pipeline :
- L'ancienne version retournait directement un objet `Annotation`
- La nouvelle version retourne un objet `DiarizeOutput` qui contient `Annotation` dans son attribut `.diarization`

### Le fix

```python
# Détecte automatiquement le type de retour
if hasattr(diarization_output, 'diarization'):
    # Nouvelle API : extraire l'Annotation
    diarization = diarization_output.diarization
else:
    # Ancienne API : déjà une Annotation
    diarization = diarization_output

# Maintenant on peut utiliser itertracks()
for turn, _, speaker in diarization.itertracks(yield_label=True):
    ...
```

## 🆘 Besoin d'aide supplémentaire ?

Si après avoir suivi TOUTES ces étapes le problème persiste :

1. Exécutez : `python test_import.py` (script ci-dessus)
2. Copiez la sortie complète
3. Exécutez : `git log --oneline -5`
4. Copiez la sortie
5. Fournissez ces deux sorties pour analyse

Cela permettra de diagnostiquer exactement où se situe le problème.
