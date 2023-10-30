import os
import glob
import _pickle as cPickle
import dlib
import cv2
import numpy as np
import _pickle as cPickle
from flask import Flask, render_template, Response, send_file, redirect, url_for, request, session
from cameras import cameras



app = Flask(__name__)



cap = None

detectorFace = dlib.get_frontal_face_detector()
detectorPontos = dlib.shape_predictor("recursos/shape_predictor_68_face_landmarks.dat")
reconhecimentoFacial = dlib.face_recognition_model_v1("recursos/dlib_face_recognition_resnet_model_v1.dat")
indices = np.load("recursos/indices_rn.pickle", allow_pickle=True)
descritoresFaciais = np.load("recursos/descritores_rn.npy")
limiar = 0.5



def detect_faces():
    global cap
    if cap is not None and cap.isOpened():
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

                cv2.rectangle(frame, (e, t), (d, b), ((128, 210, 41)
                                                       ), 2)


                texto = str("{}".format(nome)).upper().split('-')[0]

                cv2.putText(frame, texto, (e, t-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (128, 212, 41), thickness=1)

            if not validacao:
                break

            ret, jpeg = cv2.imencode('.jpg', frame)
            frame =  jpeg.tobytes()
            yield (b' --frame\r\n'
                  b' Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/modules')
def modules():
    return render_template('modules.html')


@app.route('/reconhecimento', methods=['GET'])
def reconhecimento():
    global cap
    camera = request.args.get('camera')

    if camera:
        release_camera()
        if camera == 'Portarias':
            cap = cv2.VideoCapture(0)
        elif camera == 'Oficina':
            cap = cv2.VideoCapture(1)

        return render_template('reconhecimento.html', camera_name=camera)
    else:
        return "Parâmetro 'camera' não especificado na URL."


@app.route('/resultados')
def resultados():
    return render_template('resultados.html')
@app.route('/video_feed')
def video_feed():
    return Response(detect_faces(), mimetype='multipart/x-mixed-replace;boundary=frame')

def release_camera():
    global cap
    if cap is not None:
        cap.release()
        cap = None


if __name__ == '__main__':
    app.run(debug=True)

