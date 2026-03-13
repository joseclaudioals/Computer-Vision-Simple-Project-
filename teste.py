import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image, paste_image

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))
caminho_teste = os.path.join(DIRETORIO_ATUAL, "imagens", "img.png")

if not os.path.exists(caminho_teste):
    print(f"❌ ERRO: O arquivo não foi encontrado em: {caminho_teste}")
    sys.exit()

with open(caminho_teste, "rb") as f:
    array_bytes = np.frombuffer(f.read(), dtype=np.uint8)
cv_img = cv2.imdecode(array_bytes, cv2.IMREAD_UNCHANGED)

if cv_img is None:
    print("\n❌ ERRO CRÍTICO: Imagem vazia ou corrompida.")
    sys.exit()

# O fundo deve sempre ter 3 canais (BGR)
if len(cv_img.shape) == 3 and cv_img.shape[2] == 4:
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2BGR)

# O MediaPipe EXIGE RGB
rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)

detection_result = detector.detect(img)

# ---> CORREÇÃO DE CORES <---
# Usamos a cv_img original (BGR) como tela de desenho, em vez da RGB do MediaPipe
fundo_para_desenhar = cv_img.copy()

imagem_com_sticker = paste_image(fundo_para_desenhar, detection_result)
annotated_image = draw_landmarks_on_image(imagem_com_sticker, detection_result)

cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
# Como já desenhamos na BGR, apenas mostramos a tela!
cv2.imshow("Tracking", annotated_image)

print("✅ Tudo certo! Pressione qualquer tecla na janela da imagem para fechar...")
cv2.waitKey(0)
cv2.destroyAllWindows()