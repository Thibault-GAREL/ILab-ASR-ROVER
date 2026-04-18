#!/bin/bash
# Script d'installation des dépendances pour le système ASR multilingue
# Usage: bash install_dependencies.sh

set -e  # Arrêter en cas d'erreur

echo "======================================================================="
echo "Installation des dépendances - Système ASR Multilingue avec ROVER"
echo "======================================================================="

# Couleurs pour l'affichage
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

# Vérifier Python
echo ""
echo "[1/6] Vérification de Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_info "Python trouvé: $PYTHON_VERSION"
else
    log_error "Python 3 n'est pas installé"
    exit 1
fi

# Vérifier pip
echo ""
echo "[2/6] Vérification de pip..."
if command -v pip &> /dev/null || command -v pip3 &> /dev/null; then
    PIP_CMD=$(command -v pip3 || command -v pip)
    log_info "pip trouvé: $PIP_CMD"
else
    log_error "pip n'est pas installé"
    exit 1
fi

# Mise à jour de pip
echo ""
echo "[3/6] Mise à jour de pip..."
$PIP_CMD install --upgrade pip
log_info "pip mis à jour"

# Installation des dépendances de base
echo ""
echo "[4/6] Installation des dépendances de base..."
log_warn "Cela peut prendre 10-15 minutes..."

$PIP_CMD install numpy soundfile python-dotenv PyYAML

log_info "Dépendances de base installées"

# Installation de PyTorch
echo ""
echo "[5/6] Installation de PyTorch..."
log_warn "PyTorch est volumineux (~2GB), cela peut prendre du temps..."

# Détecter si GPU est disponible
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU NVIDIA détecté - Installation de PyTorch avec support CUDA"
    $PIP_CMD install torch torchaudio
else
    log_info "Pas de GPU détecté - Installation de PyTorch CPU"
    $PIP_CMD install torch torchaudio
fi

log_info "PyTorch installé"

# Installation des bibliothèques ASR
echo ""
echo "[6/6] Installation des bibliothèques ASR et diarisation..."

$PIP_CMD install faster-whisper pyannote.audio huggingface-hub

log_info "Bibliothèques ASR installées"

# Vérification finale
echo ""
echo "======================================================================="
echo "Vérification de l'installation..."
echo "======================================================================="

python3 << 'EOF'
import sys

packages = {
    'torch': 'PyTorch',
    'torchaudio': 'TorchAudio',
    'faster_whisper': 'Faster Whisper',
    'pyannote.audio': 'Pyannote.audio',
    'numpy': 'NumPy',
    'soundfile': 'SoundFile'
}

all_ok = True
for module, name in packages.items():
    try:
        __import__(module.replace('-', '_').replace('.', '_'))
        print(f"✓ {name}")
    except ImportError:
        print(f"✗ {name} - MANQUANT")
        all_ok = False

if all_ok:
    print("\n✓ Toutes les dépendances sont installées correctement!")
    sys.exit(0)
else:
    print("\n✗ Certaines dépendances sont manquantes")
    sys.exit(1)
EOF

VERIFY_EXIT=$?

if [ $VERIFY_EXIT -eq 0 ]; then
    echo ""
    echo "======================================================================="
    log_info "INSTALLATION RÉUSSIE!"
    echo "======================================================================="
    echo ""
    echo "Prochaines étapes:"
    echo "  1. Configurez votre token HuggingFace: python setup_token.py"
    echo "  2. Générez un fichier audio de test: python generate_test_audio.py"
    echo "  3. Testez le système: python test_diarization_fix.py"
    echo ""
else
    echo ""
    echo "======================================================================="
    log_error "INSTALLATION INCOMPLÈTE"
    echo "======================================================================="
    echo ""
    echo "Certaines dépendances n'ont pas été installées correctement."
    echo "Essayez de réinstaller manuellement avec:"
    echo "  pip install -r requirements-whisper-only.txt"
    echo ""
    exit 1
fi
