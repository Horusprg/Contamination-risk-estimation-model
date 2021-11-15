#Imports do projeto

import torch
from datetime import datetime, time
from enum import auto
from os import name
from re import X

import dash
from dash.html.Frame import Frame
from dash import dcc
from dash import html
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
import streamlink

cont_hist = [0]
segundo = [0]
dado = []
videoconfig = []

#Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6")
model = torch.hub.load('yolov5', 'custom', path='wheight/YoloV5s6-v3.pt', source='local')
model.multi_label = False
model.conf = 0.3
model.iou = 0

#URL do vídeo de stream
url = "https://youtu.be/046c2M3azCg"
streams = streamlink.streams(url)

class VideoCamera(object):
    def __init__(self):
        #Escolhe a melhor qualidade de vídeo
        self.video = cv.VideoCapture(streams["best"].url)
        #Contadores de frames
        self.frames = 0
        self.count = 0
        dado.append([[0, 0, 0, 0, 1],[int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)),  -int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)), 0, 0, 1]])

    def __del__(self):
        self.video.release()

    def get_frame(self, segundo, cont_hist, dado):

        ok, image = self.video.read()
        detect = model(image, size = 2560)
        peoples = detect.pandas().xyxy[0]
        peoples = peoples.to_numpy()
        posicoes = []
        for people in peoples:
            xmin, ymin, xmax, ymax, confidence, label = int(people[0]), int(people[1]), int(people[2]), int(people[3]), people[4], people[6]
            horC = int((xmin + xmax)/2)
            verC = int((ymin + ymax)/2)
            cv.putText(image, str(confidence),(xmax + 20,ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
            cv.putText(image, label,(xmax + 20,ymin+20), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
            self.count+=1
            posicoes.append([horC,verC, xmax-xmin, ymin])
        
        #Cria dois laços para detectar distancia entre pessoas proximas
        for j in range(len(posicoes)):
            aux = float("inf")
            for i in range(len(posicoes)):
                Xmed = (posicoes[i][2]+posicoes[j][2])/2
                dhor = (posicoes[i][0]-posicoes[j][0])**2
                dver = (posicoes[i][1]-posicoes[j][1])**2
                dist = (sqrt(dhor+dver))*0.15/Xmed
                if dist < aux and dist != 0:
                    aux = dist
            if aux == float("inf"):
                posicoes[j].append(0)
            else:
                posicoes[j].append(aux)

        #Define o circulo de avaliação de perigo
            #até 2 metros
            if aux < 2:
                color = (0,0,255)
            #entre 2 a 4 metros
            elif aux < 4 and aux >= 2:
                color = (0,255,255)
            #maior que 4 metros
            if aux >= 4 or aux == 0:
                color = (0,255,0)
                        
            cv.circle(image, (posicoes[j][0], posicoes[j][3]), 5, (color), -1)
            cv.putText(image, str(float("{0:.2f}".format(posicoes[j][4])))+" m", (int(posicoes[j][0]+posicoes[j][2]/2)+20, posicoes[j][3] + 40), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                    color, 1, cv.LINE_AA)

        #implementa a contagem de pessoas e self. para gerar a media.
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
        box = bboxes[indice]
        x,y,w,h  = box[0], box[1], box[2], box[3]
        posicoes.append([x,y,w,h])
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
    app.run_server(debug=False)
    