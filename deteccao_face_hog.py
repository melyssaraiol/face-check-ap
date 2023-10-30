import cv2
import dlib

imagem = cv2.imread("fotos/vale.png")
detector = dlib.get_frontal_face_detector()
facesDetectadas = detector(imagem)
print(facesDetectadas)
print("Faces detectadas", len(facesDetectadas))
for face in facesDetectadas:
    e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
    cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 0), 2)

cv2.imshow("Detector hog", imagem)
cv2.waitKey(0)

cv2.destroyAllWindows()