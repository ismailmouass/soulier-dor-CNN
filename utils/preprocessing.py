# ============================================================
#  utils/preprocessing.py — Détection faciale & prétraitement
# ============================================================

import os
import shutil
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN

from config import DATASET_PATH, DATASET_FACES, IMG_SIZE


detector = MTCNN()


def extraire_visage(img_path: str, target_size: tuple = IMG_SIZE) -> Image.Image | None:
    """
    Détecte et recadre le visage principal dans une image.
    Retourne l'image entière redimensionnée si aucun visage n'est trouvé.

    Args:
        img_path    : chemin vers l'image source
        target_size : taille de sortie (largeur, hauteur)

    Returns:
        Image PIL redimensionnée, ou None si lecture impossible
    """
    img = cv2.imread(img_path)
    if img is None:
        return None

    img_rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resultats = detector.detect_faces(img_rgb)

    if resultats:
        best        = max(resultats, key=lambda x: x['confidence'])
        x, y, w, h  = best['box']
        marge       = int(0.35 * max(w, h))
        x1 = max(0, x - marge)
        y1 = max(0, y - marge)
        x2 = min(img_rgb.shape[1], x + w + marge)
        y2 = min(img_rgb.shape[0], y + h + marge)
        region = img_rgb[y1:y2, x1:x2]
    else:
        region = img_rgb

    return Image.fromarray(region).resize(target_size)


def extraire_region_np(image_np: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Détecte le visage dans un tableau numpy (utilisé pour l'inférence Gradio).

    Returns:
        (region_np, visage_detecte) — region recadrée et flag booléen
    """
    resultats = detector.detect_faces(image_np)
    if resultats:
        best        = max(resultats, key=lambda x: x['confidence'])
        x, y, w, h  = best['box']
        marge       = int(0.35 * max(w, h))
        x1 = max(0, x - marge)
        y1 = max(0, y - marge)
        x2 = min(image_np.shape[1], x + w + marge)
        y2 = min(image_np.shape[0], y + h + marge)
        return image_np[y1:y2, x1:x2], True
    return image_np, False


def creer_dataset_visages(classes: list[str]) -> None:
    """
    Parcourt tout le dataset brut, applique la détection faciale
    et sauvegarde les visages recadrés dans DATASET_FACES.

    Args:
        classes : liste des noms de dossiers (un par joueur)
    """
    if os.path.exists(DATASET_FACES):
        shutil.rmtree(DATASET_FACES)
    os.makedirs(DATASET_FACES)

    total_ok, total_skip = 0, 0

    for cls in classes:
        src  = os.path.join(DATASET_PATH, cls)
        dst  = os.path.join(DATASET_FACES, cls)
        os.makedirs(dst, exist_ok=True)
        imgs = os.listdir(src)
        print(f'  {cls:<30} ({len(imgs)} imgs)...', end=' ')
        ok = 0
        for img_file in imgs:
            try:
                visage = extraire_visage(os.path.join(src, img_file))
                if visage:
                    visage.save(os.path.join(dst, img_file), 'JPEG', quality=90)
                    ok += 1
            except Exception:
                total_skip += 1
        total_ok += ok
        print(f'✓ {ok}')

    print(f'\n✅ Dataset visages créé : {total_ok} images | Ignorées : {total_skip}')
