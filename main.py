#Imports do projeto
import torch
import dash
import dash_core_components as dcc
import dash_html_components as html
kernel = None
from flask import Flask, Response
import cv2 as cv
from math import sqrt, e
from datetime import timedelta
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
from dash.dependencies import Output, Input
import streamlink
import time

#Variáveis de dados
cont_hist = [0]
timer = [0]
dado = []
videoconfig = []
risk = [0,0]
local_length, local_width, local_height = 0, 0, 0
total_capacity = 20
labels = [["with_mask", "withou_mask", "mask_weared_incorrectly"],[0,0,0]]

#Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6") disponível na pasta /wheight
model = torch.hub.load('yolov5', 'custom', path='wheight/YoloV5s6.pt', source='local')
model.conf = 0.3
model.iou = 0.20

#URL do vídeo de stream
url = "https://youtu.be/U2qwkqgLYAw"
streams = streamlink.streams(url)

#Contamination risk estimation model
def air_flow_rate(Cs):
        Q = (5.2)/(Cs - 419)
        return Q

def quanta_concentration(q,Q, time, Volume):
    qc = (q/Q)*(1 - e*(-(Q*time)/Volume))
    return qc

def infection_prob(q,Q, time):
    P = (1.0 - pow(e,(-(q*0.016*time)/Q/60)))
    return P

def risk_rate(P, time, qc):
    R = 100 * (1 - pow(e,(-P*time*qc/60)))
    return R

#Detecção de vídeo
class VideoCamera(object):
    def __init__(self):
        #Escolhe a melhor qualidade de vídeo
        self.video = cv.VideoCapture(streams["best"].url)
        #Contadores de frames
        self.tinit = time.time()
        self.frames = 0
        self.count = 0
        dado.append([[int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)),int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)), 0, 0, 0], [0,0, 0, 0, 0]])

    def __del__(self):
        self.video.release()

    def get_frame(self, timer, cont_hist, dado, with_mask=0,prev_frame_time=0):

        ok, image = self.video.read()
        detect = model(image, size = 1080)
        peoples = detect.pandas().xyxy[0]
        peoples = peoples.to_numpy()
        posicoes = []
        for people in peoples:
            xmin, ymin, xmax, ymax, confidence, label = int(people[0]), int(people[1]), int(people[2]), int(people[3]), people[4], people[6]
            horC = int((xmin + xmax)/2)
            verC = int((ymin + ymax)/2)
            cv.putText(image, str(confidence),(xmax + 20,ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
            cv.putText(image, label,(xmax + 20,ymin+30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
            self.count+=1
            posicoes.append([horC,verC, xmax-xmin, ymin])
            labels.append(label)
            if label == 'with_mask':
                with_mask += 1
                labels[1][0]+=1
            
            if label == 'without_mask':
                labels[1][1]+=1
            
            if label == 'mask_weared_incorrectly':
                labels[1][2]+=1
            
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
                        
            cv.circle(image, (posicoes[j][0], posicoes[j][3]), 7, (color), -1)
            cv.putText(image, str(float("{0:.2f}".format(posicoes[j][4])))+" m", (int(posicoes[j][0]+posicoes[j][2]/2)+20, posicoes[j][3] + 60), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                    color, 1, cv.LINE_AA)

        #implementa a contagem de pessoas e self. para gerar a media.
        self.frames += 1
        pessoas = int(self.count / self.frames)

        #Volume de um escritório padrão
        volume = 9.29*2.70*total_capacity
        #volume = local_height*local_width*local_height
            
        #Imprime o contador de média de pessoas detectadas por frame
        cv.putText(image, str(pessoas) + " Pessoas", (100, 80),
                cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
            # quantos self. por frame
        
        new_frame_time = time.time()
        fps = 1/(new_frame_time-prev_frame_time)
        prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv.putText(image, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)
        dado.append(posicoes)
        if self.frames == 30:
            self.frames = 0
            tempo = time.time()-self.tinit
            cont_hist.append(pessoas)
            timer.append(tempo)
            self.count = 0
            #Área de Cálculo de risco
            if pessoas != 0:
                Q = pessoas*air_flow_rate(439)
                q = 2.3666*(0.4+0.6*(pessoas-with_mask)/(pessoas))
                qc = quanta_concentration(q, Q, ((tempo)), volume)
                P = infection_prob(q, Q, ((tempo)))
                R = risk_rate(P,((tempo)/60), qc)
                risk.append(R)

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

#Stream da detecção de vídeo
def gen(camera):
    while True:
        frame = camera.get_frame(timer, cont_hist, dado)
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
    for sec in timer:
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
            zAxes.append(cont_hist)
    fig = go.Figure(go.Histogram2d(
                    x=(xAxes),
                    y=(yAxes),
                    z=(zAxes)
    ))
    return fig

@app.callback(
            Output('live-velocimeter', 'figure'),
            Input('interval-component', 'n_intervals')
            )
def velocimeter(n):
    fig = go.Figure()
    fig.add_trace(go.Indicator(
    value = risk[-1],
    delta = {'reference': risk[-2]},
    gauge = {
        'axis': {'visible': False}},
    domain = {'row': 0, 'column': 0}))
    fig.update_layout(
    grid = {'rows': 1, 'columns': 1, 'pattern': "independent"},
    template = {'data' : {'indicator': [{
        'number':{'font_color':"white", 'suffix': "%"},
        'gauge':{'axis_range': (0,100)},
        'title': {'text': "Risco de contaminação", 'font_color':"white", 'font_size': 48},
        'mode' : "number+delta+gauge",
        'delta' : {'reference': risk[-2]}}]
                         }})
    fig.update_layout(paper_bgcolor = "rgb(3, 7, 15)")
    return fig

@app.callback(
            Output('live-pie', 'figure'),
            Input('interval-component', 'n_intervals')
            )

def pie(n):
    fig = px.pie(values = labels[1], names = labels[0])
    fig.update_layout(legend_font_size = 32,paper_bgcolor = "rgb(3, 7, 15)")
    
    return fig

app.layout = html.Div(
    className= "layout",
    children=[
        html.H5("MONITORAMENTO DE LOCAL", className="anim-typewriter"),
        html.Img(className="button",src="assets/Group 3.png"),
        html.H4("AMBIENTE"),
        html.Img(className= "video",src="/video_feed"),
        html.Div("CLASSES", className="classes"),
        html.H3("PESSOAS AO LONGO DO DIA", className="contPess"),
        dcc.Graph(id='live-update-graph', className='contagem'),
        html.H3("MAPA DE OCUPAÇÃO", className="contPess2d"),
        dcc.Graph(id='live-update-3d', className='contagem3d'),
        dcc.Graph(id='live-velocimeter', className='velocimeter'),
        dcc.Graph(id='live-pie', className='pie'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,
            n_intervals=0
            )
    ]
)

if __name__ == '__main__':
    app.run_server(debug=False)
    