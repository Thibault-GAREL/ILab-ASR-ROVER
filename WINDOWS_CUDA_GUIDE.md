# Guide CUDA/cuDNN pour Windows

## 🔴 Erreur Commune sur Windows

Si vous voyez cette erreur :
```
Could not locate cudnn_ops64_9.dll. Please make sure it is in your library path!
Invalid handle. Cannot load symbol cudnnCreateTensorDescriptor
```

**Cause** : Le système essaie d'utiliser CUDA/GPU mais cuDNN n'est pas installé correctement sur Windows.

## ✅ Solution Rapide : Utiliser le CPU

### Option 1 : Utiliser la config CPU (Recommandé pour Windows)

```bash
# Utiliser la config Windows optimisée
python examples/cli_transcribe.py audio.wav --whisper-only --config configs/config-windows-cpu.yaml
```

### Option 2 : Modifier la config par défaut

J'ai déjà modifié `configs/config.yaml` pour utiliser CPU. Relancez simplement :

```powershell
python examples/cli_transcribe.py data/input/test_meeting.wav --whisper-only
```

## 🚀 Pour Utiliser le GPU (Avancé)

Si vous voulez vraiment utiliser votre GPU NVIDIA sur Windows, vous devez installer CUDA et cuDNN correctement :

### Prérequis

1. **GPU NVIDIA** compatible (série GTX/RTX)
2. **Windows 10/11** 64-bit
3. **Pilotes NVIDIA** à jour
4. **~8GB d'espace disque** pour CUDA + cuDNN

### Étape 1 : Vérifier votre GPU

```powershell
nvidia-smi
```

Si cette commande fonctionne, vous avez un GPU NVIDIA. Notez la version CUDA supportée.

### Étape 2 : Installer CUDA Toolkit

1. **Télécharger CUDA 12.x** :
   - https://developer.nvidia.com/cuda-downloads
   - Choisissez : Windows → x86_64 → Version de Windows → exe (network)

2. **Installer** :
   - Exécutez l'installeur
   - Choisissez "Custom Installation"
   - Cochez au minimum :
     - CUDA Runtime
     - CUDA Development
     - Visual Studio Integration (si vous avez VS)

3. **Vérifier l'installation** :
   ```powershell
   nvcc --version
   ```

### Étape 3 : Installer cuDNN

1. **Créer un compte NVIDIA Developer** (gratuit) :
   - https://developer.nvidia.com/

2. **Télécharger cuDNN** :
   - https://developer.nvidia.com/cudnn
   - Choisissez la version compatible avec votre CUDA (ex: cuDNN 9.x pour CUDA 12.x)
   - Format : ZIP pour Windows

3. **Installer cuDNN** :
   ```powershell
   # Extraire le ZIP téléchargé
   # Copier les fichiers dans le dossier CUDA :

   # Dossier CUDA par défaut : C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\

   # Copier :
   # - cudnn*/bin/*.dll → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin\
   # - cudnn*/include/*.h → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\
   # - cudnn*/lib/x64/*.lib → C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\lib\x64\
   ```

   Ou via PowerShell (nécessite Admin) :
   ```powershell
   # Remplacer les chemins selon votre installation
   $cudnn_path = "C:\Downloads\cudnn-windows-x86_64-9.x.x"
   $cuda_path = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"

   Copy-Item "$cudnn_path\bin\*.dll" -Destination "$cuda_path\bin\" -Force
   Copy-Item "$cudnn_path\include\*.h" -Destination "$cuda_path\include\" -Force
   Copy-Item "$cudnn_path\lib\x64\*.lib" -Destination "$cuda_path\lib\x64\" -Force
   ```

4. **Ajouter au PATH** :
   ```powershell
   # Ajouter ces chemins aux variables d'environnement :
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\libnvvp
   ```

   Via l'interface Windows :
   - Paramètres Windows → Système → Paramètres système avancés
   - Variables d'environnement
   - Dans "Variables système", éditer "Path"
   - Ajouter les chemins CUDA

### Étape 4 : Réinstaller PyTorch avec Support CUDA

```powershell
# Désinstaller PyTorch CPU
pip uninstall torch torchaudio -y

# Installer PyTorch avec CUDA 12.x
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Étape 5 : Vérifier l'Installation

```python
python -c "import torch; print(f'CUDA disponible: {torch.cuda.is_available()}'); print(f'Nombre de GPUs: {torch.cuda.device_count()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Résultat attendu** :
```
CUDA disponible: True
Nombre de GPUs: 1
GPU: NVIDIA GeForce RTX 3060 (ou votre modèle)
```

### Étape 6 : Modifier la Config pour GPU

```yaml
# Dans configs/config.yaml
general:
  device: "cuda"

diarization:
  device: "cuda"

whisper:
  device: "cuda"
  compute_type: "float16"  # Plus rapide sur GPU

canary:
  device: "cuda"
```

### Étape 7 : Tester

```powershell
python examples/cli_transcribe.py audio.wav --whisper-only
```

## 📊 Performances Attendues

### Mode CPU (config actuelle)

| Durée Audio | Temps de Traitement | Vitesse |
|-------------|---------------------|---------|
| 1 minute    | ~2-4 minutes        | 2-4x    |
| 10 minutes  | ~20-40 minutes      | 2-4x    |
| 1 heure     | ~2-4 heures         | 2-4x    |

**Modèle recommandé** : `medium` (bon compromis vitesse/qualité)

### Mode GPU (avec CUDA/cuDNN)

| Durée Audio | Temps de Traitement | Vitesse |
|-------------|---------------------|---------|
| 1 minute    | ~10-20 secondes     | 0.2x    |
| 10 minutes  | ~2-3 minutes        | 0.2x    |
| 1 heure     | ~12-20 minutes      | 0.2x    |

**Modèle recommandé** : `large-v3` (meilleure qualité)

## 🔍 Dépannage

### "CUDA out of memory"
- Réduisez la taille du modèle : `medium` au lieu de `large-v3`
- Réduisez `batch_size` dans la config
- Fermez les autres applications utilisant le GPU

### "RuntimeError: CUDA error: no kernel image is available"
- Version de PyTorch incompatible avec votre GPU
- Réinstallez PyTorch avec la bonne version CUDA

### "torch.cuda.is_available() retourne False"
- Vérifiez que vous avez installé PyTorch avec CUDA : `pip show torch`
- Réinstallez avec `--index-url https://download.pytorch.org/whl/cu124`

### cuDNN toujours introuvable après installation
1. Vérifiez que les DLL sont dans `CUDA\vXX\bin\`
2. Redémarrez votre terminal/IDE
3. Vérifiez le PATH système
4. Essayez de copier manuellement `cudnn*.dll` dans le dossier de votre script

## 🎯 Recommandation

**Pour la plupart des utilisateurs Windows** :
- ✅ Utilisez le **mode CPU** (plus simple, fonctionne directement)
- ✅ Modèle `medium` pour un bon compromis
- ✅ Laissez tourner pendant la nuit pour les longs audios

**Si vous avez besoin de performance** :
- Installez CUDA + cuDNN (complexe mais plus rapide)
- Ou utilisez un service cloud avec GPU (Google Colab, AWS)
- Ou divisez vos audios en segments plus courts

## 📚 Ressources

- **CUDA Toolkit** : https://developer.nvidia.com/cuda-downloads
- **cuDNN** : https://developer.nvidia.com/cudnn
- **PyTorch Installation** : https://pytorch.org/get-started/locally/
- **NVIDIA Drivers** : https://www.nvidia.com/download/index.aspx

## ✅ Vérification Finale

Après avoir modifié la config pour CPU, testez :

```powershell
# Nettoyer le cache Python
Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" -Force | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# Tester avec un vrai fichier audio
python examples/cli_transcribe.py votre_audio.wav --whisper-only
```

**Vous ne devriez plus voir d'erreurs cuDNN !** 🎉
