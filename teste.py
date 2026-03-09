import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

img = mp.Image.create_from_file("imagens/img.png")

detection_result = detector.detect(img)

annotade_image = draw_landmarks_on_image(img.numpy_view(), detection_result)
cv2.imshow(cv2.cvtColor(annotade_image, cv2.COLOR_RGB2BGR))
