# GUIDE DE CONFIGURATION - ASR ROVER
# Suivez ces étapes pour que TOUT fonctionne

## ✅ ÉTAPE 1 : Créer un Token HuggingFace

1. Allez sur : https://huggingface.co/settings/tokens
2. Cliquez sur "New token"
3. Nom : `pyannote-asr`
4. Type : **Read**
5. Cliquez "Generate"
6. **COPIEZ** le token (il commence par `hf_`)

## ✅ ÉTAPE 2 : Accepter les Licences

**IMPORTANT** : Vous devez accepter les conditions d'utilisation :

1. Allez sur : https://huggingface.co/pyannote/speaker-diarization-3.1
2. Cliquez sur **"Agree and access repository"**
3. Allez sur : https://huggingface.co/pyannote/segmentation-3.0
4. Cliquez sur **"Agree and access repository"**

## ✅ ÉTAPE 3 : Configurer le Token dans .env

### Option A : Via Notepad (Recommandé)

```powershell
# Créer le fichier
copy .env.example .env

# Ouvrir avec Notepad
notepad .env
```

Dans Notepad, modifiez la ligne :
```
HF_TOKEN=your_huggingface_token_here
```

En :
```
HF_TOKEN=hf_VotrEtOkEnIcI
```

**Sauvegardez** (Ctrl+S) et fermez Notepad.

### Option B : Via PowerShell

```powershell
# Créer .env avec le token directement
@"
HF_TOKEN=hf_VotrEtOkEnIcI
DEVICE=cpu
"@ | Out-File -Encoding utf8 .env
```

## ✅ ÉTAPE 4 : Vérifier la Configuration

```powershell
python setup_token.py
```

Vous devriez voir :
- ✓ Fichier .env trouvé
- ✓ HF_TOKEN configuré dans .env
- ✓ Token valide trouvé: hf_xxxxxxx...
- ✓ Connexion réussie à HuggingFace!

## ✅ ÉTAPE 5 : Tester le Système

### Test 1 : Whisper seul (sans diarisation)

```powershell
# Générer un fichier audio de test
python generate_test_audio.py

# Tester Whisper seul
python test_whisper_seul.py
```

### Test 2 : Système complet (avec diarisation)

```powershell
# Avec votre fichier
python examples/cli_transcribe.py data/input/meeting.wav --whisper-only

# Ou avec le fichier de test
python examples/cli_transcribe.py data/input/test_meeting.wav --whisper-only
```

## ❌ DÉPANNAGE

### Problème 1 : "401 Client Error: Unauthorized"

**Solution :**
1. Vérifiez que le token est dans .env : `type .env`
2. Vérifiez qu'il commence par `hf_`
3. Acceptez les licences (ÉTAPE 2 ci-dessus)
4. Relancez : `python setup_token.py`

### Problème 2 : "No HuggingFace token provided"

**Solution :**
```powershell
# Vérifier si .env existe
dir .env

# Si non, créer :
copy .env.example .env
notepad .env

# Vérifier le contenu
type .env
```

### Problème 3 : "File not found: meeting.wav"

**Solution :**
```powershell
# Générer un fichier de test
python generate_test_audio.py

# Vérifier qu'il existe
dir data\input\test_meeting.wav

# Utiliser ce fichier
python examples/cli_transcribe.py data/input/test_meeting.wav --whisper-only
```

### Problème 4 : python-dotenv pas installé

**Solution :**
```powershell
pip install python-dotenv
```

## 📋 CHECKLIST COMPLÈTE

Avant de lancer le système, vérifiez :

- [ ] Token HF créé sur huggingface.co
- [ ] Licences acceptées pour pyannote/speaker-diarization-3.1
- [ ] Licences acceptées pour pyannote/segmentation-3.0
- [ ] Fichier .env créé avec le token
- [ ] python-dotenv installé
- [ ] Fichier audio disponible (ou généré)
- [ ] `python setup_token.py` affiche tous les ✓

## 🚀 COMMANDES RAPIDES (Copy-Paste)

```powershell
# Configuration complète
copy .env.example .env
notepad .env  # Ajoutez votre token HF_TOKEN=hf_...

# Vérification
python setup_token.py

# Génération audio de test
python generate_test_audio.py

# Test Whisper seul
python test_whisper_seul.py

# Test complet avec diarisation
python examples/cli_transcribe.py data/input/test_meeting.wav --whisper-only
```

## 💡 IMPORTANT

Le flag `--whisper-only` signifie :
- ✅ Utilise Whisper pour la transcription
- ✅ Utilise Pyannote pour la diarisation (identification speakers)
- ❌ N'utilise PAS Canary/NeMo (évite les conflits de dépendances)

Pour la diarisation, vous **DEVEZ** avoir le token HF configuré !

## 📞 BESOIN D'AIDE ?

Si après avoir suivi TOUTES ces étapes ça ne fonctionne toujours pas :

1. Lancez : `python setup_token.py` et copiez la sortie
2. Lancez : `type .env` et vérifiez le contenu (masquez le token)
3. Indiquez à quelle étape exactement ça bloque
