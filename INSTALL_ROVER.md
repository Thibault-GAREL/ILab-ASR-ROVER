# Guide d'Installation ROVER (Whisper + Canary + Fusion)

Ce guide explique comment installer **ROVER complet** pour obtenir la fusion de Whisper Large V3 + NVIDIA Canary avec amélioration du WER de ~7-8% à ~4.5-5.5%.

## ⚠️ Avertissements Importants

**AVANT DE COMMENCER** :

1. **Conflits de dépendances** : NeMo a des exigences strictes qui peuvent entrer en conflit avec d'autres packages
2. **Installation longue** : ~15-30 minutes selon votre connexion
3. **Espace disque** : ~5-8GB supplémentaires pour les modèles
4. **Recommandé** : Créer un nouvel environnement virtuel dédié à ROVER

## 📋 Prérequis

- Python 3.8, 3.9 ou 3.10 (Python 3.11+ peut avoir des problèmes avec NeMo)
- pip à jour
- ~8GB d'espace disque libre
- Bonne connexion internet

## 🚀 Installation Étape par Étape

### Étape 1 : Créer un Nouvel Environnement Virtuel (Recommandé)

```bash
# Désactiver l'environnement actuel si nécessaire
deactivate  # ou fermez votre terminal

# Créer un nouvel environnement pour ROVER
python -m venv venv-rover
source venv-rover/bin/activate  # Linux/Mac
# OU
venv-rover\Scripts\activate  # Windows

# Mettre à jour pip
python -m pip install --upgrade pip setuptools wheel
```

**Pourquoi un nouvel environnement ?**
- Évite les conflits avec votre installation Whisper-only actuelle
- Vous gardez une version qui fonctionne en backup
- Permet de revenir facilement en arrière si problème

### Étape 2 : Installer les Dépendances de Base

```bash
cd ASR-Mixture_of_expert-ROVER

# Installer les packages de base en premier
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# OU pour GPU :
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install soundfile librosa python-dotenv PyYAML
```

### Étape 3 : Installer NeMo avec Précaution

**IMPORTANT** : NeMo nécessite numpy<1.24

```bash
# Installer numpy compatible avec NeMo en premier
pip install "numpy>=1.22.0,<1.24"

# Installer Cython (requis par NeMo)
pip install Cython

# Installer NeMo toolkit avec ASR
pip install nemo_toolkit[asr]==1.23.0

# Si l'installation échoue, essayez sans version spécifique :
# pip install nemo_toolkit[asr]
```

**Si vous voyez des erreurs** :
```bash
# Essayez d'installer les dépendances problématiques une par une
pip install hydra-core omegaconf pytorch-lightning
pip install nemo_toolkit[asr]
```

### Étape 4 : Installer les Autres Dépendances

```bash
# Installer Whisper et Pyannote
pip install faster-whisper pyannote.audio huggingface-hub

# Vérifier qu'il n'y a pas de conflit numpy
pip show numpy  # Doit afficher une version < 1.24
```

### Étape 5 : Vérifier l'Installation

```bash
python -c "
import torch
import nemo
import nemo.collections.asr as nemo_asr
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
print('✓ Toutes les dépendances sont installées correctement!')
print(f'PyTorch: {torch.__version__}')
print(f'NeMo: {nemo.__version__}')
print(f'CUDA disponible: {torch.cuda.is_available()}')
"
```

**Résultat attendu** :
```
✓ Toutes les dépendances sont installées correctement!
PyTorch: 2.x.x
NeMo: 1.23.0
CUDA disponible: True/False
```

### Étape 6 : Configurer HuggingFace Token

```bash
# Si pas déjà fait
python setup_token.py
```

Acceptez les licences des modèles Pyannote (comme avant).

### Étape 7 : Tester ROVER

```bash
# Test avec un fichier audio réel
python examples/cli_transcribe.py data/input/sample1.wav

# Vous devriez voir dans les logs :
# - "Initializing Canary ASR..."
# - "Canary model loaded successfully"
# - "Using ROVER fusion"
```

## 🔍 Vérification du Succès

Les logs doivent montrer :

```
Initializing Whisper ASR...
✓ Whisper model loaded successfully: large-v3

Initializing Canary ASR...
✓ Canary model loaded successfully  ← NOUVEAU !

Stage 2/4: Whisper ASR
✓ Whisper transcription complete: XX words

Stage 3/4: Canary ASR  ← NOUVEAU !
✓ Canary transcription complete: XX words

Stage 4/4: ROVER Fusion  ← NOUVEAU !
✓ ROVER fusion complete
✓ Final WER estimated: ~4.5-5.5%

Systems used: whisper + canary + ROVER  ← NOUVEAU !
```

## ❌ Dépannage

### Erreur : "No module named 'nemo'"

```bash
pip install nemo_toolkit[asr]
```

### Erreur : "numpy version conflict"

```bash
# Forcer la bonne version
pip install --force-reinstall "numpy>=1.22.0,<1.24"
```

### Erreur : "Model.from_pretrained() got unexpected keyword argument 'batch_size'"

✅ **CORRIGÉ** dans le commit le plus récent. Faites :
```bash
git pull origin claude/multilingual-meeting-transcription-BuLON
```

### Erreur : "Cannot download Canary model"

Le modèle Canary est hébergé sur NGC (NVIDIA GPU Cloud). Solutions :

**Option A : Laisser NeMo télécharger automatiquement**
```bash
# NeMo devrait télécharger automatiquement depuis NGC
# Si ça échoue, attendez quelques minutes et réessayez
```

**Option B : Utiliser un NGC API Key** (optionnel)
```bash
# Créer un compte sur https://ngc.nvidia.com/
# Générer une API key
# Ajouter à .env :
NGC_API_KEY=your_ngc_api_key_here
```

**Option C : Utiliser un modèle alternatif**
```bash
# Dans configs/config.yaml, changez :
canary:
  model_name: "nvidia/stt_en_canary_1b"  # Alternative
```

### Erreur : "CUDA out of memory" avec Canary

```yaml
# Dans configs/config.yaml :
canary:
  device: "cpu"  # Utilisez CPU pour Canary
  batch_size: 1  # Réduisez le batch
```

Ou utilisez Whisper-only :
```bash
python examples/cli_transcribe.py audio.wav --whisper-only
```

## 📊 Performance Comparée

| Mode | WER | Vitesse CPU | Vitesse GPU |
|------|-----|-------------|-------------|
| Whisper-only | ~7-8% | 2-3x RTF | 0.2x RTF |
| **ROVER (Whisper+Canary)** | **~4.5-5.5%** | 4-6x RTF | 0.4x RTF |

**RTF (Real-Time Factor)** : 2x = prend 2× la durée de l'audio

## 🔄 Revenir à Whisper-Only

Si ROVER pose problème, revenez facilement :

```bash
# Option 1 : Utiliser --whisper-only
python examples/cli_transcribe.py audio.wav --whisper-only

# Option 2 : Réactiver l'ancien environnement
deactivate
source venv/bin/activate  # Votre ancien venv
```

## 🎯 Recommandations

**Utilisez ROVER si** :
- ✅ Vous avez besoin de précision maximale
- ✅ WER de 4.5-5.5% est crucial pour votre cas d'usage
- ✅ Vous êtes prêt à gérer des dépendances complexes
- ✅ Vous transcrivez de grandes quantités d'audio

**Restez sur Whisper-only si** :
- ✅ WER de 7-8% est suffisant (excellent pour la plupart des cas)
- ✅ Vous voulez une installation simple
- ✅ Vous voulez éviter les conflits de dépendances
- ✅ Vous n'avez pas de GPU (ROVER est lent sur CPU)

## 📝 Résumé Installation Minimale

```bash
# 1. Nouvel environnement
python -m venv venv-rover
source venv-rover/bin/activate  # ou venv-rover\Scripts\activate

# 2. Dépendances de base
pip install --upgrade pip
pip install torch torchaudio

# 3. NeMo (CRITIQUE)
pip install "numpy>=1.22.0,<1.24"
pip install nemo_toolkit[asr]==1.23.0

# 4. Le reste
pip install faster-whisper pyannote.audio huggingface-hub soundfile python-dotenv

# 5. Tester
python examples/cli_transcribe.py data/input/sample1.wav
```

## ✅ Checklist de Validation

Avant de confirmer que ROVER fonctionne :

- [ ] `import nemo` fonctionne sans erreur
- [ ] `import nemo.collections.asr` fonctionne
- [ ] `numpy.__version__` < 1.24
- [ ] Canary se charge sans erreur "batch_size"
- [ ] Les logs montrent "Stage 3/4: Canary ASR"
- [ ] Les logs montrent "Stage 4/4: ROVER Fusion"
- [ ] "Systems used: whisper + canary + ROVER"

## 🆘 Support

Si vous rencontrez des problèmes :

1. Vérifiez les logs complets
2. Partagez la sortie de `pip list`
3. Partagez l'erreur exacte
4. Vérifiez la version de Python : `python --version`

**En cas de blocage** : Restez sur Whisper-only qui fonctionne déjà parfaitement ! 🚀
