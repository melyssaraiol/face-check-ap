import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

acuracia_lista = []
taxa_reconhecimento_lista = []

cap = cv2.VideoCapture(0)

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices_rn.pickle", allow_pickle=True)
descritoresFaciais = np.load("recursos/descritores_rn.npy")
limiar = 0.5

if cap.isOpened():
    cv2.namedWindow('Video da câmera', cv2.WINDOW_NORMAL)  # Cria uma janela
    while True:
        validacao, frame = cap.read()
        facesDetectadas = detectorFace(frame, 2)

        for face in facesDetectadas:
            e, t, d, b = (int(face.left()), int(face.top()), int(face.right()), int(face.bottom()))
            pontosFaciais = detectorPontos(frame, face)
            descritorFacial = reconhecimentoFacial.compute_face_descriptor(frame, pontosFaciais)
            listaDescritorFacial = [fd for fd in descritorFacial]
            npArrayDescritorFacial = np.asarray(listaDescritorFacial, dtype=np.float64)
            npArrayDescritorFacial = npArrayDescritorFacial[np.newaxis, :]

            distancias = np.linalg.norm(npArrayDescritorFacial - descritoresFaciais, axis=1)
            minimo = np.argmin(distancias)
            distanciaMinima = distancias[minimo]


            if distanciaMinima <= limiar:
                nome = os.path.split(indices[minimo])[1].split(".")[0]
            else:
                nome = ' '

            acuracia = 1.0 - distanciaMinima
            acuracia_lista.append(acuracia)
            if nome != ' ':
                taxa_reconhecimento_lista.append(1)
            else:
                taxa_reconhecimento_lista.append(0)


            cv2.rectangle(frame, (e, t), (d, b), ((128, 210, 41)
                                                   ), 2)

            texto = str("{}".format(nome)).upper().split('-')[0]

            cv2.putText(frame, texto, (e, t-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (128, 212, 41), thickness=1)

        if not validacao:
            break
        cv2.imshow("Detector hog", frame)
        key = cv2.waitKey(1)

        if key == 27:

            break

cap.release()
cv2.destroyAllWindows()

taxa_media_reconhecimento = sum(taxa_reconhecimento_lista) / len(taxa_reconhecimento_lista)
acuracia_media = sum(acuracia_lista) / len(acuracia_lista)
print(f'Taxa Média de Reconhecimento: {taxa_media_reconhecimento}')
print(f'Acurácia Média: {acuracia_media}')

# Gráfico de barras da taxa de reconhecimento ao longo do tempo
plt.figure(figsize=(10, 6))
plt.plot(range(len(taxa_reconhecimento_lista)), taxa_reconhecimento_lista)
plt.xlabel('Tempo')
plt.ylabel('Taxa de Reconhecimento')
plt.title('Taxa de Reconhecimento ao Longo do Tempo')
plt.show()

# Gráfico de barras da acurácia ao longo do tempo
plt.figure(figsize=(10, 6))
plt.plot(range(len(acuracia_lista)), acuracia_lista)
plt.xlabel('Tempo')
plt.ylabel('Acurácia')
plt.title('Acurácia ao Longo do Tempo')
plt.show()