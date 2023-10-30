import cv2

webcam = cv2.VideoCapture("rtsp://backoffice:genesis@172.20.4.58/cam/realmonitor?channel=1&subtype=0" )

if webcam.isOpened():
    cv2.namedWindow('Video da câmera', cv2.WINDOW_NORMAL)
    while True:
        validacao, frame = webcam.read()
        if not validacao:
            break
        cv2.imshow('Video da câmera', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

webcam.release()
cv2.destroyAllWindows()
