import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

while True:
    sucesso, img=cam.read()
    if not sucesso:
        break

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
