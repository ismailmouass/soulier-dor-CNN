# ============================================================
#  utils/data_loader.py — Générateurs de données & augmentation
# ============================================================

import json
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from config import DATASET_FACES, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES_PATH


def construire_generateurs() -> tuple:
    """
    Crée les générateurs train / validation avec augmentation.

    Returns:
        (train_gen, val_gen, class_names)
        - train_gen   : ImageDataGenerator enrichi (rotation, flip, zoom…)
        - val_gen     : ImageDataGenerator simple (normalisation seulement)
        - class_names : dict {index_str -> nom_joueur}
    """
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        # Augmentations géométriques
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.1,
        # Augmentations photométriques
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )
    val_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_FACES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=SEED
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_FACES,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=SEED
    )

    # Sauvegarder le mapping classes ↔ index
    class_names = {str(v): k for k, v in train_gen.class_indices.items()}
    with open(CLASS_NAMES_PATH.replace('/drive/MyDrive/soulier_dor_CNN/', '/content/'), 'w') as f:
        json.dump(class_names, f)

    print(f'Train : {train_gen.samples} images | Val : {val_gen.samples} images')
    print(f'Classes : {train_gen.num_classes}')

    return train_gen, val_gen, class_names
