import cv2

img = cv2.imread("imagens/img.png", cv2.IMREAD_UNCHANGED)

if img is None:
    raise FileExistsError("imagem nao encontrada")

cv2.imshow('img', img)

cv2.waitKey(0)

cv2.destroyAllWindows()