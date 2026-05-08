# ============================================================
#  setup_colab.py — Initialisation rapide dans Google Colab
#
#  Exécuter cette cellule en PREMIER dans Colab :
#
#  !git clone https://github.com/ismailmouass/soulier-dor-CNN.git
#  %cd soulier-dor-CNN
#  !pip install -r requirements.txt -q
#  # ⚠️ Redémarrer le runtime après pip install
#
#  Puis dans une nouvelle cellule :
#  exec(open('setup_colab.py').read())
# ============================================================

import subprocess
import sys
import os

# ── 1. Vérification GPU ────────────────────────────────────
import tensorflow as tf
print('TensorFlow :', tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'✅ GPU actif : {gpus[0].name}')
else:
    print('⚠️  Aucun GPU détecté — entraînement sera lent !')
    print('   → Colab : Exécution > Modifier le type d\'exécution > GPU T4')

# ── 2. Vérification des dépendances ───────────────────────
deps = ['mtcnn', 'gradio', 'lz4', 'cv2', 'sklearn']
manquants = []
for dep in deps:
    try:
        __import__(dep)
    except ImportError:
        manquants.append(dep)

if manquants:
    print(f'\n⚠️  Dépendances manquantes : {manquants}')
    print('   → Exécute : !pip install -r requirements.txt -q')
    print('   → Puis redémarre le runtime')
else:
    print('✅ Toutes les dépendances sont disponibles')

# ── 3. Vérification structure fichiers ────────────────────
fichiers_requis = [
    'config.py', 'train.py', 'predict.py',
    'model/cnn_model.py',
    'utils/preprocessing.py',
    'utils/data_loader.py',
]
tous_ok = True
for f in fichiers_requis:
    if os.path.exists(f):
        print(f'  ✓ {f}')
    else:
        print(f'  ✗ {f} — MANQUANT !')
        tous_ok = False

# ── 4. Rappel configuration ───────────────────────────────
print('\n' + '='*55)
if tous_ok:
    print('✅ Environnement prêt !')
    print('\nProchaines étapes :')
    print('  1. Éditer config.py → renseigner KAGGLE_USERNAME et KAGGLE_KEY')
    print('  2. Entraînement   : !python train.py')
    print('  3. Démonstration  : !python predict.py')
else:
    print('❌ Des fichiers sont manquants — re-cloner le dépôt')
    print('   !git clone https://github.com/ismailmouass/soulier-dor-CNN.git')
print('='*55)
