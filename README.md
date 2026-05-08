# 🥇 Classification de Footballeurs par CNN — Soulier d'Or

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)](https://keras.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ismailmouass/soulier-dor-CNN/blob/main/projet_final_cnn_clean.ipynb)

> Identification automatique de 22 légendes du football ayant remporté le Soulier d'Or européen,
> par réseaux de neurones convolutifs (CNN) avec Transfer Learning.

**ENSA de Fès — Module : Deep Learning & NLP — 2025/2026**
**Auteurs :** Ayman Lebbar & Ismail Mouass | **Encadrant :** Pr. Oussama EL GANNOUR

---

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Val Accuracy (Top-1) | **80,8 %** |
| Top-3 Accuracy | **91,9 %** |
| Durée entraînement | ~22 min (GPU T4) |
| Taille modèle | 27,3 MB |

---

## 🏗️ Architecture du pipeline

```
Photo brute
    ↓
MTCNN (Détection faciale)
    ↓
Recadrage 160×160
    ↓
MobileNetV2 (Transfer Learning, ImageNet)
    ↓
GlobalAveragePooling2D
    ↓
Dense(256) + BatchNorm + Dropout(0.5)
    ↓
Softmax → 22 joueurs
```

---

## 📁 Structure du projet

```
soulier-dor-CNN/
│
├── config.py                        # Paramètres globaux (chemins, hyperparamètres)
├── train.py                         # Pipeline d'entraînement complet (Phase 1 + 2)
├── predict.py                       # Interface Gradio d'inférence
├── requirements.txt                 # Dépendances Python
├── projet_final_cnn_clean.ipynb     # Notebook Colab (version tout-en-un)
│
├── model/
│   └── cnn_model.py                 # Architecture MobileNetV2 + callbacks
│
├── utils/
│   ├── preprocessing.py             # Détection faciale MTCNN
│   └── data_loader.py               # Générateurs de données & augmentation
│
└── samples/                         # Images de démonstration
```

---

## ⚙️ Lancement

### Option 1 — Google Colab (recommandé)

1. Ouvrir le notebook :

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ismailmouass/soulier-dor-CNN/blob/main/projet_final_cnn_clean.ipynb)

2. Activer le GPU : **Exécution → Modifier le type d'exécution → GPU (T4)**
3. Exécuter toutes les cellules dans l'ordre

> ⚠️ Après la Cellule 1 (installation), **redémarrer le runtime** avant de continuer.

---

### Option 2 — Fichiers Python séparés (depuis Colab)

**Étape 1 — Cloner le dépôt et installer les dépendances :**
```python
!git clone https://github.com/ismailmouass/soulier-dor-CNN.git
%cd soulier-dor-CNN
!pip install -r requirements.txt -q
```

**Étape 2 — Configurer vos identifiants Kaggle dans `config.py` :**
```python
KAGGLE_USERNAME = 'votre_username'
KAGGLE_KEY      = 'votre_cle_api'
```

**Étape 3 — Lancer l'entraînement :**
```python
!python train.py
```

**Étape 4 — Lancer l'interface de démonstration :**
```python
!python predict.py
```

---

### Option 3 — Rechargement rapide (modèle déjà entraîné sur Drive)

Si le modèle est déjà sauvegardé sur Google Drive, exécutez uniquement :

```python
# Dans Colab
!pip install lz4 mtcnn gradio -q
# Redémarrer le runtime, puis :
!python predict.py
```

---

## 📦 Dépendances

```bash
pip install -r requirements.txt
```

Ou manuellement :
```bash
pip install tensorflow>=2.15 mtcnn lz4 gradio opencv-python scikit-learn matplotlib kaggle
```

---

## 📊 Dataset

- **Source** : [Golden Foot Football Players Image Dataset](https://www.kaggle.com/datasets/balabaskar/golden-foot-football-players-image-dataset) (Kaggle)
- **Volume** : 7 190 images JPEG
- **Classes** : 22 joueurs (Messi, Ronaldo, Maradona, Pelé, Ronaldinho…)
- **Distribution** : 241 à 351 images/classe (moyenne : 326)

---

## 🧠 Détails du modèle

| Composant | Détail |
|---|---|
| Base | MobileNetV2 (ImageNet, 2,2M params gelés) |
| Tête | Dense(256) + BatchNorm + Dropout(0.5) |
| Sortie | Dense(22, Softmax) |
| Perte | CategoricalCrossentropy (label_smoothing=0.1) |
| Prétraitement | MTCNN face detection → crop 160×160 |
| Inférence | TTA ×10 + seuil confiance 40% |

---

## 🔄 Protocole d'entraînement

**Phase 1** — Base gelée (15 époques, LR = 1e-3)
- Seule la tête de classification est entraînée
- Val Accuracy : ~40%

**Phase 2** — Fine-tuning (30 époques, LR = 5e-5)
- Les 54 dernières couches de MobileNetV2 sont dégelées
- Val Accuracy : **80,8%** (+41 points)

---

## 🎮 Interface de démonstration (Gradio)

- Upload d'une photo de footballeur
- Détection automatique du visage (MTCNN)
- Prédiction avec Top-5 et barre de confiance
- Refus automatique si confiance < 40%

---

## 📄 Licence

Projet académique — ENSA de Fès 2025/2026.
Dataset : [Kaggle Golden Foot](https://www.kaggle.com/datasets/balabaskar/golden-foot-football-players-image-dataset)
