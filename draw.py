import cv2
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
    # Faz uma cópia da matriz da imagem original
    annotated_image = np.copy(rgb_image)
    altura, largura, _ = annotated_image.shape

    # Pega a lista de mãos detectadas pela IA
    hand_landmarks_list = detection_result.hand_landmarks
    
    # Se não tem mão na foto, devolve a imagem normal
    if not hand_landmarks_list:
        return annotated_image

    # Mapeamento manual de quais pontos se conectam na mão (os "ossos")
    conexoes = [
        (0, 1), (1, 2), (2, 3), (3, 4),         # Dedão
        (0, 5), (5, 6), (6, 7), (7, 8),         # Indicador
        (5, 9), (9, 10), (10, 11), (11, 12),    # Dedo Médio
        (9, 13), (13, 14), (14, 15), (15, 16),  # Anelar
        (13, 17), (0, 17), (17, 18), (18, 19), (19, 20) # Mindinho
    ]

    for hand_landmarks in hand_landmarks_list:
        # 1. Desenha as linhas verdes (ossos)
        for conexao in conexoes:
            # Pega as coordenadas percentuais (0.0 a 1.0)
            pt1 = hand_landmarks[conexao[0]]
            pt2 = hand_landmarks[conexao[1]]
            
            # Multiplica pela largura/altura para achar o pixel exato na tela
            x1, y1 = int(pt1.x * largura), int(pt1.y * altura)
            x2, y2 = int(pt2.x * largura), int(pt2.y * altura)
            
            # Traça a linha na matriz da imagem
            cv2.line(annotated_image, (x1, y1), (x2, y2), (88, 205, 54), 2)
            
        # 2. Desenha as bolinhas vermelhas (juntas)
        for landmark in hand_landmarks:
            x, y = int(landmark.x * largura), int(landmark.y * altura)
            cv2.circle(annotated_image, (x, y), 4, (0, 0, 255), -1)

    return annotated_image