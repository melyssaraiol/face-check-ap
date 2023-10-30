import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
from PIL import Image

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices_yale.pickle", allow_pickle=True)
descritoresFaciais = np.load("recursos/descritores_yale.npy")
limiar = 0.5
totalFaces = 0
totalAcertos = 0

imagens = glob.glob(os.path.join("yalefaces/teste", "*.gif"))
for arquivo in imagens:
    imagemFace = Image.open(arquivo).convert('RGB')
    imagem = np.array(imagemFace, 'uint8')
    idAtual = int(os.path.split(arquivo)[1].split('.')[0].replace('subject', ''))
    totalFaces += 1
    facesDetectadas = detectorFace(imagem, 2)
    for face in facesDetectadas:
        e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
        pontosFaciais = detectorPontos(imagem, face)
        descritorFacial = reconhecimentoFacial.compute_face_descriptor(imagem, pontosFaciais)
        listaDescritorFacial = [fd for fd in descritorFacial]
        npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
        npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

        distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
        #print("Distâncias: {}".format(distancias))
        minimo = np.argmin(distancias)
        #print(minimo)
        distanciaMinima = distancias[minimo]
        #print(distanciaMinima)

        if distanciaMinima <= limiar:
            nome = os.path.split(indices[minimo])[1].split(".")[0]
            idPrevisto = int(os.path.split(indices[minimo])[1].split('.')[0].replace('subject', ''))
            if idPrevisto == idAtual:
                totalAcertos += 1
        else:
            nome = 'Desconhecido'


        print(f'idAtual: {idAtual} idPrevisto: {idPrevisto}')
        cv2.rectangle(imagem, (e, t), (d, b), (0, 255, 255), 2)
        texto = "{} {:.4f}".format(nome, distanciaMinima)
        cv2.putText(imagem, texto, (d, t), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 255))

    cv2.imshow("Detector hog", imagem)
    cv2.waitKey(0)

percentualAcerto = (totalAcertos / totalFaces) * 100
print(f'Percentual de acertos: {percentualAcerto}')
cv2.destroyAllWindows()