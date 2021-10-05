from datetime import datetime, time
from enum import auto
from os import name
from re import X

import dash
from dash.html.Frame import Frame
import dash_core_components as dcc
import dash_html_components as html
kernel = None

from flask import Flask, Response
import cv2 as cv
from math import sqrt
from datetime import datetime, timedelta
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from dash.dependencies import Output, Input
import plotly

classes = []
with open("yolo/coco.names", "rt") as f:
    classes = f.read().rstrip("\n").split("\n")

cont_hist = [0]
segundo = [0]
dado = []
videoconfig = []
#habilitando o net
net = cv.dnn.readNetFromDarknet("yolo/yolov3.cfg", "yolo/yolov3.weights")
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


class VideoCamera(object):
    def __init__(self):
        self.video = cv.VideoCapture("video/walking.avi")
        self.frames = 0
        self.count = 0
        dado.append([[0, int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)), 0, 0, 1],[int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)), 0, 0, 0, 1]])

    def __del__(self):
        self.video.release()

    def get_frame(self, segundo, cont_hist, dado):
        success, image = self.video.read()
        blob = cv.dnn.blobFromImage(image, 0.1/255, (320,320), [0,0,0], 1, crop = False)
        net.setInput(blob)

        layerNames = net.getLayerNames()
        outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
        outputs = net.forward(outputNames)

        posicoes = []
        #encontra os objetos na imagem tratada
        contornos = findObjects(outputs, image)
        for contorno in contornos:
            if contorno[4] == "person":
                x, y, l, a = contorno[0], contorno[1], contorno[2], contorno[3]
                horC = int(x + l/2)
                verC = int(y + a/2)
                cv.putText(image, 'Pessoa',(x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv.LINE_AA)
                self.count+=1
                posicoes.append([horC,verC,int(l/2),int(a/2)])
        
        #Cria dois laços para detectar distancia entre pessoas proximas
        for j in range(len(posicoes)):
            aux = float("inf")
            for i in range(len(posicoes)):
                dist = sqrt((posicoes[i][0] - posicoes[j][0]) ** 2) + (((posicoes[i][1]+posicoes[i][3]) - (posicoes[j][1]+posicoes[j][3])) ** 2)
                if dist < aux and dist != 0:
                    aux = dist
            posicoes[j].append(int(aux))

        #Define o circulo de avaliação de perigo
            #até 2 metros
            if aux < (posicoes[j][0]+posicoes[j][2])*2:
                cv.ellipse(image,(posicoes[j][0],posicoes[j][1]+posicoes[j][3]),(25,5),0,0,360,(0,0,255),2)
                cv.circle(image, (posicoes[j][0], posicoes[j][1]), 7, (0, 0, 255), -1)
                cv.line(image, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0],posicoes[j][1]+posicoes[j][3] ), (0, 0, 255), 2)

                # Exibe o objeto a menor distancia encontrado da referencia
                cv.putText(image, str(float("{0:.2f}".format(aux/(posicoes[j][0]+posicoes[j][2]))))+" m", (posicoes[j][0] + 40, posicoes[j][1]+posicoes[j][3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 0, 255), 1, cv.LINE_AA)

            #entre 2 a 4 metros
            elif aux > (posicoes[j][0]+posicoes[j][2])*2-1 and aux < (posicoes[j][0]+posicoes[j][2])*4:
                cv.ellipse(image,(posicoes[j][0],posicoes[j][1]+posicoes[j][3]),(25,5),0,0,360,(0,255,255),2)
                cv.circle(image, (posicoes[j][0], posicoes[j][1]), 7, (0, 255, 255), -1)
                cv.line(image, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0], posicoes[j][1]+posicoes[j][3]), (0, 255, 255), 2)

                # Exibe o objeto a menor distancia encontrado da referencia
                cv.putText(image, str(float("{0:.2f}".format(aux/(posicoes[j][0]+posicoes[j][2]))))+" m", (posicoes[j][0] + 40, posicoes[j][1]+posicoes[j][3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 255), 1, cv.LINE_AA)
            #maior que 4 metros
            if aux > (posicoes[j][0]+posicoes[j][2])*4-1:
                cv.ellipse(image,(posicoes[j][0],posicoes[j][1]+posicoes[j][3]),(25,5),0,0,360,(0,255,0),2)
                cv.circle(image, (posicoes[j][0], posicoes[j][1]), 7, (0, 255, 0), -1)
                cv.line(image, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0], posicoes[j][1]+posicoes[j][3]), (0, 255, 0), 2)

                # Exibe o objeto a menor distancia encontrado da referencia
                cv.putText(image, str(float("{0:.2f}".format(aux/(posicoes[j][0]+posicoes[j][2]))))+" m", (posicoes[j][0] + 40, posicoes[j][1]+posicoes[j][3] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                        (0, 255, 0), 1, cv.LINE_AA)

        #implementa a contagem de pessoas e self. para gerar a media
        self.frames += 1
        pessoas = int(self.count / self.frames)

        #Imprime o contador de média de pessoas detectadas por frame
        cv.putText(image, str(pessoas) + " Pessoas", (100, 80),
                cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
            # quantos self. por frame
        dado.append(posicoes)
        if self.frames == 10:
            self.frames = 0
            cont_hist.append(pessoas)
            segundo.append(segundo[-1]+1)
            self.count = 0

        ret, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()

def findObjects(outputs,frame):
    hT, wT, cT = frame.shape
    bboxes = []
    classIds = []
    confs = []
    posicoes = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]

            if confidence > 0.5:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT) - w/2) , int((det[1]*hT) - h/2)
                bboxes.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    indices = cv.dnn.NMSBoxes(bboxes, confs, 0.5, 0.3)

    for indice in indices:
        indice = indice[0]
        box = bboxes[indice]
        x,y,w,h,id  = box[0], box[1], box[2], box[3], classes[classIds[indice]]
        posicoes.append([x,y,w,h,id])
    return posicoes


def gen(camera):
    while True:
        frame = camera.get_frame(segundo, cont_hist, dado)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')

def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.callback(
            Output('live-update-graph', 'figure'),
            Input('interval-component', 'n_intervals')
            )

def update_graph(n):
    tempo = []
    for sec in segundo:
        sec = str(timedelta(seconds = sec))
        tempo.append(sec)

    fig = go.Figure(layout={"template":"plotly_dark"})
    fig.add_trace(go.Bar(x=tempo, y=cont_hist))
    fig.update_layout(
        paper_bgcolor="#242424",
        plot_bgcolor="#242424",
        autosize=True,
        margin=dict(l=10, r=10, b=30, t=10),
        )

    return  fig

@app.callback(
            Output('live-update-3d', 'figure'),
            Input('interval-component', 'n_intervals')
            )

def update_3d(n):
    xAxes = []
    yAxes = []
    zAxes = []
    for i in range(0, len(dado)):
        for j in range(0, len(dado[i])):
            xAxes.append(dado[i][j][0])
            yAxes.append(-dado[i][j][1])
            zAxes.append(dado[i][j][4])
    fig = go.Figure(go.Histogram2d(
                    x=(xAxes),
                    y=(yAxes),
                    z=(zAxes)
    ))
    return fig
        
app.layout = html.Div(
    className= "layout",
    children=[
        html.H5("MONITORAMENTO DE LOCAL", className="anim-typewriter"),
        html.Img(className="button",src="assets/Group 3.png"),
        html.H4("AMBIENTE"),
        html.Img(className= "video",src="/video_feed"),
        html.Div("NÚMERO DE PESSOAS", className="contagem-left"),
        html.Div("LIMITE PERMITIDO", className="contagem-right"),
        html.H3("PESSOAS AO LONGO DO DIA", className="contPess"),
        dcc.Graph(id='live-update-graph', className='contagem'),
        html.H3("MAPA DE OCUPAÇÃO", className="contPess2d"),
        dcc.Graph(id='live-update-3d', className='contagem3d'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
            )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=True)
    