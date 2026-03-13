import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from draw import paste_image

base_options = python.BaseOptions(model_asset_path = "hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options = base_options, num_hands = 2)
detector = vision.HandLandmarker.create_from_options(options)

cam = cv2.VideoCapture(0)

while True:
    sucesso, frame = cam.read()
    if not sucesso:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame_rgb)

    resultado = detector.detect(mp_image)

    frame_final = paste_image(frame, resultado)

    cv2.imshow("tracker", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cam.relese()
cv2.destroyAllWindows()
