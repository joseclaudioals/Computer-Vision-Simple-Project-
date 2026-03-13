import cv2
import numpy as np
import os

DIRETORIO_ATUAL = os.path.dirname(os.path.abspath(__file__))

def carregar_imagem(nome_arquivo):
    caminho_completo = os.path.join(DIRETORIO_ATUAL, "imagens", nome_arquivo)
    
    if not os.path.exists(caminho_completo):
        print(f"⚠️ AVISO: A imagem não foi encontrada em: {caminho_completo}")
        return None
    
    try:
        with open(caminho_completo, "rb") as f:
            array = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(array, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"⚠️ Erro ao tentar ler {caminho_completo}: {e}")
        return None

# Carrega as imagens originais (com fundo transparente)
img_esq = carregar_imagem("seven.png")
img_dir = carregar_imagem("six.png")

# --- ALTERAÇÃO DE TAMANHO E PROPORÇÃO ---
# Em vez de um tamanho único, definimos largura e altura separadas.
# Aumentamos um pouco o tamanho geral e deixamos a altura maior que a largura.
LARGURA_STICKER = 120  # Aumentamos de 100 para 120
ALTURA_STICKER = 150   # Aumentamos de 100 para 150 (ficando mais alto do que largo)

# Redimensiona as imagens se elas existirem
if img_esq is not None:
    img_esq = cv2.resize(img_esq, (LARGURA_STICKER, ALTURA_STICKER))
if img_dir is not None:
    img_dir = cv2.resize(img_dir, (LARGURA_STICKER, ALTURA_STICKER))
# ----------------------------------------

def colar_com_transparencia(fundo, overlay, x, y):
    bg_h, bg_w, _ = fundo.shape
    h, w = overlay.shape[0], overlay.shape[1]

    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return fundo

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    ox1, oy1 = max(0, -x), max(0, -y)
    ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)

    if overlay.shape[2] == 4:
        alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
        alpha = np.dstack((alpha, alpha, alpha))
        cor_overlay = overlay[oy1:oy2, ox1:ox2, :3]
        fundo[y1:y2, x1:x2] = (alpha * cor_overlay + (1 - alpha) * fundo[y1:y2, x1:x2]).astype(np.uint8)
    else:
        fundo[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2]

    return fundo

def draw_landmarks_on_image(rgb_image, detection_result):
    annotated_image = np.copy(rgb_image)
    altura, largura, _ = annotated_image.shape
    
    if not detection_result.hand_landmarks:
        return annotated_image

    conexoes = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
    ]

    for hand_landmarks in detection_result.hand_landmarks:
        for conexao in conexoes:
            pt1 = hand_landmarks[conexao[0]]
            pt2 = hand_landmarks[conexao[1]]
            x1, y1 = int(pt1.x * largura), int(pt1.y * altura)
            x2, y2 = int(pt2.x * largura), int(pt2.y * altura)
            cv2.line(annotated_image, (x1, y1), (x2, y2), (88, 205, 54), 2)
            
        for landmark in hand_landmarks:
            x, y = int(landmark.x * largura), int(landmark.y * altura)
            cv2.circle(annotated_image, (x, y), 4, (0, 0, 255), -1)

    return annotated_image

def paste_image(fundo_imagem, detection_result):
    annotated_image = fundo_imagem.copy()
    altura_tela, largura_tela, _ = annotated_image.shape

    if detection_result.hand_landmarks:
        for i in range(len(detection_result.hand_landmarks)):
            mao = detection_result.hand_landmarks[i]
            tipo_mao = detection_result.handedness[i][0].category_name

            img_atual = img_esq if tipo_mao == "Left" else img_dir
            if img_atual is None:
                continue

            # Matematica pra centralizar na palma (média dos 4 pontos centralizados)
            soma_x = mao[0].x + mao[5].x + mao[9].x + mao[13].x
            soma_y = mao[0].y + mao[5].y + mao[9].y + mao[13].y
            
            centro_x = int((soma_x / 4) * largura_tela)
            centro_y = int((soma_y / 4) * altura_tela)

            # Ajustar os bgl foda (posições)
            metade_largura = LARGURA_STICKER // 2
            metade_altura = ALTURA_STICKER // 2
            
            inicio_x = centro_x - metade_largura
            inicio_y = centro_y - metade_altura
            # -------------------------

            annotated_image = colar_com_transparencia(annotated_image, img_atual, inicio_x, inicio_y)

    return annotated_image