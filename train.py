# ============================================================
#  train.py — Pipeline d'entraînement complet (Phase 1 + Phase 2)
#  Exécuter dans Google Colab avec GPU activé
# ============================================================

# ── Étape 1 : Imports ──────────────────────────────────────
import os
import json
import zipfile
import warnings

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

warnings.filterwarnings('ignore')
tf.random.set_seed(42)

from config import (
    DATASET_PATH, DATASET_FACES, DRIVE_PATH,
    KAGGLE_USERNAME, KAGGLE_KEY,
    P1_EPOCHS, P2_EPOCHS, SEED
)
from utils.preprocessing import creer_dataset_visages
from utils.data_loader import construire_generateurs
from model.cnn_model import (
    construire_modele, compiler_phase1,
    debloquer_fine_tuning, get_callbacks, afficher_resume
)

print('TensorFlow :', tf.__version__)
print('GPU actif  :', tf.config.list_physical_devices('GPU'))


# ── Étape 2 : Téléchargement du dataset Kaggle ─────────────
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump({'username': KAGGLE_USERNAME, 'key': KAGGLE_KEY}, f)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

os.system('kaggle datasets download -d balabaskar/golden-foot-football-players-image-dataset -p /content/')

with zipfile.ZipFile('/content/golden-foot-football-players-image-dataset.zip', 'r') as z:
    z.extractall('/content/golden_foot_dataset')

print('✅ Dataset téléchargé !')


# ── Étape 3 : Détection faciale MTCNN ──────────────────────
classes = sorted(os.listdir(DATASET_PATH))
print(f'\n{len(classes)} joueurs trouvés — création du dataset visages...')
creer_dataset_visages(classes)


# ── Étape 4 : Générateurs de données ───────────────────────
train_gen, val_gen, class_names = construire_generateurs()


# ── Étape 5 : Construction du modèle ───────────────────────
model, base = construire_modele()
compiler_phase1(model)
afficher_resume(model)


# ── Étape 6 : Phase 1 — Base gelée ─────────────────────────
print('\n' + '=' * 55)
print('  PHASE 1 — Base gelée (LR=1e-3)')
print('=' * 55)

h1 = model.fit(
    train_gen,
    epochs=P1_EPOCHS,
    validation_data=val_gen,
    callbacks=get_callbacks('phase1_best'),
    verbose=1
)

best_p1 = max(h1.history['val_accuracy'])
print(f'\n✅ Meilleure val_accuracy Phase 1 : {best_p1:.4f}')


# ── Étape 7 : Phase 2 — Fine-tuning ────────────────────────
print('\n' + '=' * 55)
print('  PHASE 2 — Fine-tuning (LR=5e-5)')
print('=' * 55)

debloquer_fine_tuning(model, base)

h2 = model.fit(
    train_gen,
    epochs=P2_EPOCHS,
    validation_data=val_gen,
    callbacks=get_callbacks('phase2_best'),
    verbose=1
)

best_p2 = max(h2.history['val_accuracy'])
print(f'\n✅ Meilleure val_accuracy Phase 2 : {best_p2:.4f}')
print(f'   Gain vs Phase 1               : +{best_p2 - best_p1:.4f}')


# ── Étape 8 : Courbes d'apprentissage ──────────────────────
def merge_history(h1, h2):
    return {k: h1.history[k] + h2.history[k] for k in h1.history}

hall = merge_history(h1, h2)
sep  = len(h1.history['accuracy'])
ep   = range(1, len(hall['accuracy']) + 1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Courbes d'apprentissage — CNN Soulier d'Or", fontsize=14, fontweight='bold')

for ax, (tk, vk, titre) in zip(axes, [
    ('accuracy',  'val_accuracy',  'Accuracy'),
    ('loss',      'val_loss',      'Loss'),
    ('top3_acc',  'val_top3_acc',  'Top-3 Accuracy'),
]):
    ax.plot(ep, hall[tk], label='Train', color='#2196F3', linewidth=2)
    ax.plot(ep, hall[vk], label='Val',   color='#FF9800', linewidth=2)
    ax.axvline(x=sep, color='red', linestyle='--', linewidth=1.5, label='Fine-tuning')
    ax.set_title(titre)
    ax.set_xlabel('Époque')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/content/courbes_apprentissage.png', dpi=150, bbox_inches='tight')
plt.show()
print(f'val_accuracy : {best_p2:.4f} | val_top3_acc : {max(h2.history["val_top3_acc"]):.4f}')


# ── Étape 9 : Matrice de confusion ─────────────────────────
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

val_gen.reset()
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = val_gen.classes
noms_classes = [k.replace('_', ' ').title() for k in sorted(train_gen.class_indices.keys())]

cm  = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(16, 14))
ConfusionMatrixDisplay(cm, display_labels=noms_classes).plot(ax=ax, cmap='Blues', xticks_rotation=45)
ax.set_title("Matrice de confusion — CNN Soulier d'Or", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/content/matrice_confusion.png', dpi=150, bbox_inches='tight')
plt.show()


# ── Étape 10 : Sauvegarde sur Google Drive ─────────────────
import shutil
from google.colab import drive

drive.mount('/drive')
os.makedirs(DRIVE_PATH, exist_ok=True)

fichiers = {
    '/content/checkpoints/phase2_best.keras' : 'phase2_best.keras',
    '/content/class_names.json'               : 'class_names.json',
    '/content/courbes_apprentissage.png'       : 'courbes_apprentissage.png',
    '/content/matrice_confusion.png'           : 'matrice_confusion.png',
}

for src, dst in fichiers.items():
    if os.path.exists(src):
        shutil.copy(src, f'{DRIVE_PATH}/{dst}')
        print(f'  ✓ {dst}')

print('\n✅ Tout sauvegardé sur Google Drive !')
print(f'📁 Chemin : {DRIVE_PATH}')
