# 🥇 Classification de Footballeurs par CNN — Soulier d'Or

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green)](https://keras.io)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

> Identification automatique de 22 légendes du football ayant remporté le Soulier d'Or européen,
> par réseaux de neurones convolutifs (CNN) avec Transfer Learning.

---

## 📋 Description du projet

Ce projet implémente un pipeline CNN complet de classification d'images pour identifier
automatiquement 22 footballeurs légendaires (Messi, Ronaldo, Maradona, Ronaldinho, Pelé...)
à partir de photographies.

**Résultats obtenus :**
| Métrique | Valeur |
|---|---|
| Val Accuracy (Top-1) | **80,8 %** |
| Top-3 Accuracy | **91,9 %** |
| Durée entraînement | ~22 min (GPU T4) |
| Taille modèle | 27,3 MB |

---

## 🏗️ Architecture du pipeline
