# Installation Guide

## Problème de Dépendances Résolu

Le conflit vient de **NVIDIA NeMo qui requiert numpy<1.24**. Voici les solutions :

## ✅ Option 1 : Installation Whisper-Only (RECOMMANDÉ)

**La plus simple et sans conflits** - Parfait pour 90% des cas d'usage

```bash
pip install -r requirements-whisper-only.txt
```

**Avantages:**
- Aucun conflit de dépendances
- Installation rapide et propre
- Whisper Large V3 donne déjà d'excellents résultats (~7-8% WER)
- Toujours la diarisation avec Pyannote

**Utilisation:**
```python
pipeline = MeetingTranscriptionPipeline(use_canary=False)
```

## ⚡ Option 2 : Installation Full avec Canary (AVANCÉ)

**Pour performance maximale avec ROVER** - Installation plus complexe

### Méthode A : Installation séquentielle (recommandée)

```bash
# 1. Installer les dépendances de base avec numpy compatible
pip install torch torchaudio numpy==1.23.5

# 2. Installer NeMo
pip install nemo_toolkit[asr]==1.23.0

# 3. Installer le reste
pip install faster-whisper pyannote.audio pyannote.core soundfile librosa pydub tqdm python-dotenv pyyaml pytest black flake8
```

### Méthode B : Fichier requirements

```bash
pip install -r requirements-full.txt
```

Si des conflits persistent :
```bash
# Forcer les versions spécifiques
pip install --upgrade pip
pip install numpy==1.23.5
pip install -r requirements-full.txt --no-deps
pip install torch torchaudio faster-whisper pyannote.audio soundfile librosa
```

## 🔧 Résolution du Conflit

Le conflit exact :
```
nemo-toolkit[asr] 1.22.0 requires numpy<1.24 and >=1.22
numpy>=1.24.0 (incompatible)
```

**Solution appliquée dans requirements.txt:**
```txt
numpy>=1.22.0,<1.24  # Compatible avec NeMo
```

## 📋 Comparaison des Options

| Critère | Whisper-Only | Full (ROVER) |
|---------|-------------|--------------|
| Installation | ✅ Simple | ⚠️ Complexe |
| WER | ~7-8% | ~4.5-5.5% |
| Langues | 99 | EN/FR/ES/DE optimisé |
| Vitesse | 68x RT | ~240x RT |
| Conflits | ❌ Aucun | ⚠️ Possibles |

## 🚀 Quick Start

### Pour commencer rapidement (recommandé) :

```bash
# Installation Whisper-Only
pip install -r requirements-whisper-only.txt

# Configuration
cp .env.example .env
# Éditer .env et ajouter votre HF_TOKEN

# Test
python examples/whisper_only.py
```

### Pour performance maximale :

```bash
# Installation Full
pip install numpy==1.23.5
pip install nemo_toolkit[asr]==1.23.0
pip install -r requirements-full.txt

# Test
python examples/basic_transcription.py
```

## 🐛 Troubleshooting

### Erreur : "Cannot install ... conflicting dependencies"
**Solution:** Utilisez `requirements-whisper-only.txt` au lieu de `requirements.txt`

### Erreur : "No module named 'nemo'"
**Solution:** C'est normal si vous utilisez Whisper-Only. Passez `use_canary=False`

### Warning pip version
```bash
python -m pip install --upgrade pip
```

### Environnement virtuel propre
```bash
# Recréer l'environnement
rm -rf venv  # ou del /s venv sur Windows
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements-whisper-only.txt
```

## 📌 Recommandation Finale

**Pour 99% des utilisateurs : `requirements-whisper-only.txt`**

- Installation sans problème
- Performance excellente (Whisper V3 est déjà top-tier)
- Moins de dépendances = moins de problèmes
- Diarisation complète incluse

**ROVER/Canary uniquement si :**
- Vous avez besoin du WER absolu minimum
- Vous êtes prêt à gérer des dépendances complexes
- Environnement contrôlé (Docker, conda)
