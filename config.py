# ============================================================
#  config.py — Paramètres globaux du projet CNN Soulier d'Or
# ============================================================

# Chemins
DATASET_PATH   = '/content/golden_foot_dataset/football_golden_foot/football_golden_foot'
DATASET_FACES  = '/content/golden_foot_faces'
CHECKPOINTS    = '/content/checkpoints'
DRIVE_PATH     = '/drive/MyDrive/soulier_dor_CNN'

# Chemins fichiers sauvegardés
MODEL_PATH       = f'{DRIVE_PATH}/phase2_best.keras'
CLASS_NAMES_PATH = f'{DRIVE_PATH}/class_names.json'

# Paramètres modèle
IMG_SIZE    = (160, 160)
BATCH_SIZE  = 64
NUM_CLASSES = 22
SEED        = 42

# Kaggle (à remplacer par vos identifiants)
KAGGLE_USERNAME = 'VOTRE_USERNAME_KAGGLE'
KAGGLE_KEY      = 'VOTRE_CLE_KAGGLE'

# Inférence
SEUIL_CONFIANCE = 0.40
TTA_N           = 10   # nombre d'augmentations TTA

# Entraînement Phase 1
P1_EPOCHS = 15
P1_LR     = 1e-3

# Entraînement Phase 2 (fine-tuning)
P2_EPOCHS        = 30
P2_LR            = 5e-5
P2_FREEZE_LAYERS = 100   # geler les 100 premières couches de MobileNetV2
