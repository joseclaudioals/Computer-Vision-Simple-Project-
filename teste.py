import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import draw_landmarks_on_image

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

img = mp.Image.create_from_file("imagens/img1.jpg")

detection_result = detector.detect(img)

annotated_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL) # Linha que permite redimensionar imagem teste
cv2.imshow("Tracking", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

cv2.waitKey(0)
cv2.destroyAllWindows()