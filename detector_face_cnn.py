import cv2
import dlib

imagem = cv2.imread('fotos/melyssa.jpeg')
detector = dlib.cnn_face_detection_model_v1('recursos/mmod_human_face_detector.dat')
facesDetectadas = detector(imagem, 1)
print(facesDetectadas)
print('Faces detectadas: ', len(facesDetectadas))

for face in facesDetectadas:
    e, t, d, b, c = (int(face.rect.left()), int(face.rect.top()), int(face.rect.right()), int(face.rect.bottom()), face.confidence)
    print(c)
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

cv2.imshow('Detector CNN', imagem)
cv2.waitKey(0)
cv2.destroyAllWindows()