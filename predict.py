# ============================================================
#  predict.py — Interface Gradio (inférence + TTA + seuil)
#  Utiliser après entraînement (ou rechargement depuis Drive)
# ============================================================

import json
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import gradio as gr

from config import MODEL_PATH, CLASS_NAMES_PATH, SEUIL_CONFIANCE, TTA_N, IMG_SIZE
from utils.preprocessing import extraire_region_np

# ── Chargement modèle & classes ────────────────────────────
# Monter Drive si nécessaire (Colab)
try:
    from google.colab import drive
    drive.mount('/drive')
except Exception:
    pass

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH) as f:
    class_names = json.load(f)

noms = [class_names[str(i)].replace('_', ' ').title() for i in range(len(class_names))]
print(f'✅ Modèle chargé — {len(noms)} joueurs')
print(f'GPU : {tf.config.list_physical_devices("GPU")}')


# ── Test Time Augmentation ──────────────────────────────────
def tta_predict(region_np: np.ndarray, n: int = TTA_N) -> np.ndarray:
    """
    Moyenne de N prédictions légèrement augmentées (TTA).

    Args:
        region_np : image numpy RGB recadrée sur le visage
        n         : nombre d'augmentations

    Returns:
        vecteur de probabilités moyenné (shape: NUM_CLASSES)
    """
    np.random.seed(42)
    aug = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.1,
        brightness_range=[0.9, 1.1],
        preprocessing_function=preprocess_input
    )
    img_arr = np.expand_dims(
        np.array(Image.fromarray(region_np).resize(IMG_SIZE), dtype=np.float32), 0
    )
    preds = [model.predict(preprocess_input(img_arr.copy()), verbose=0)[0]]
    gen   = aug.flow(img_arr, batch_size=1, seed=42)
    for _ in range(n - 1):
        preds.append(model.predict(next(gen), verbose=0)[0])
    return np.mean(preds, axis=0)


# ── Fonction de prédiction principale ──────────────────────
def predire(image: np.ndarray) -> tuple[str, dict]:
    """
    Pipeline complet : détection visage → TTA → résultat formaté.

    Args:
        image : image numpy RGB (depuis Gradio)

    Returns:
        (texte_resultat, dict_top5_probabilites)
    """
    img_rgb        = np.array(Image.fromarray(image).convert('RGB'))
    region, trouve = extraire_region_np(img_rgb)
    note           = '✅ Visage détecté' if trouve else '⚠️ Image entière utilisée'

    pred = tta_predict(region, n=TTA_N)
    top5 = np.argsort(pred)[::-1][:5]
    conf = pred[top5[0]]

    if conf < SEUIL_CONFIANCE:
        texte = (f'{note}\n\n❓ Joueur non identifié avec certitude\n'
                 f'Confiance max : {conf * 100:.1f}%\n\nCandidats :\n')
        for i, idx in enumerate(top5[:3]):
            texte += f'  {i+1}. {noms[idx]:<25} {pred[idx]*100:.1f}%\n'
    else:
        texte  = f'{note}\n\n🏆 {noms[top5[0]]}\n'
        texte += f'Confiance : {conf * 100:.1f}%\n\nTop 5 :\n'
        for i, idx in enumerate(top5):
            texte += f'{i+1}. {noms[idx]:<25} {pred[idx]*100:.1f}%\n'

    return texte, {noms[idx]: float(pred[idx]) for idx in top5}


# ── Interface Gradio ────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🥇 Classificateur CNN — Soulier d'Or
    ### Détection faciale automatique (MTCNN) + Test Time Augmentation (TTA ×10)
    Uploadez une photo d'un footballeur parmi les **22 légendes** du Soulier d'Or !
    """)

    with gr.Row():
        img_in = gr.Image(label='📸 Photo du joueur', type='numpy', height=350)
        with gr.Column():
            txt_out   = gr.Textbox(label='🎯 Résultat de la prédiction', lines=14)
            chart_out = gr.Label(label='📊 Probabilités Top-5', num_top_classes=5)

    gr.Button('🔍 Identifier le joueur', variant='primary', size='lg').click(
        predire, inputs=img_in, outputs=[txt_out, chart_out]
    )

    gr.Markdown(
        '**Modèle :** MobileNetV2 | **Val Accuracy :** 80,8% | **Top-3 Acc :** 91,9% | '
        'ENSA Fès — Deep Learning 2025/2026'
    )

demo.launch(share=True)
