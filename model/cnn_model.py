# ============================================================
#  model/cnn_model.py — Architecture MobileNetV2 + Transfer Learning
# ============================================================

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config import IMG_SIZE, NUM_CLASSES, P1_LR, P2_LR, P2_FREEZE_LAYERS, CHECKPOINTS


def construire_modele() -> tuple[tf.keras.Model, MobileNetV2]:
    """
    Construit le modèle CNN complet :
      - Base : MobileNetV2 pré-entraîné sur ImageNet (gelée pour la Phase 1)
      - Tête : Dense(256) + BatchNorm + Dropout(0.5) + Softmax(22)

    Returns:
        (model, base) — modèle complet et référence à la base (pour le fine-tuning)
    """
    # Base pré-entraînée (gelée)
    base = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    base.trainable = False

    # Tête de classification
    inputs  = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, kernel_regularizer=regularizers.l2(0.005))(x)
    x       = layers.BatchNormalization()(x)
    x       = layers.Activation('relu')(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def compiler_phase1(model: tf.keras.Model) -> None:
    """Compile le modèle pour la Phase 1 (base gelée, LR élevé)."""
    model.compile(
        optimizer=Adam(learning_rate=P1_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )


def debloquer_fine_tuning(model: tf.keras.Model, base: MobileNetV2) -> None:
    """
    Active le fine-tuning : débloque les dernières couches de MobileNetV2
    et recompile avec un LR réduit.
    """
    base.trainable = True
    for layer in base.layers[:P2_FREEZE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=P2_LR),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')]
    )

    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f'Paramètres entraînables après déblocage : {trainable:,}')


def get_callbacks(nom: str) -> list:
    """
    Retourne les callbacks standard :
      - ModelCheckpoint (meilleur val_accuracy)
      - EarlyStopping (patience=8)
      - ReduceLROnPlateau (patience=3, factor=0.3)
    """
    import os
    os.makedirs(CHECKPOINTS, exist_ok=True)
    return [
        ModelCheckpoint(
            f'{CHECKPOINTS}/{nom}.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-8,
            verbose=1
        )
    ]


def afficher_resume(model: tf.keras.Model) -> None:
    """Affiche le nombre de paramètres total / entraînable / gelé."""
    total     = model.count_params()
    trainable = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f'Paramètres total        : {total:,}')
    print(f'Paramètres entraînables : {trainable:,}')
    print(f'Paramètres gelés        : {total - trainable:,}')
    model.summary()
