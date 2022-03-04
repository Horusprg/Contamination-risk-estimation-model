#Imports do projeto
import cv2 as cv
from math import e
import time
from numpy import size
from sqlalchemy import null
import streamlink
import torch
from datetime import timedelta
from tkinter import *
from PIL import Image, ImageTk
import matplotlib.pyplot as plt


# Contamination risk estimation model
def air_flow_rate(cs):
    q = 5.2 / (cs - 419)
    return q


def quanta_concentration(q, Q, time, Volume):
    qc = (q / Q) * (1 - e * (-(Q * time) / Volume))
    return qc   


def infection_prob(q, Q, time):
    P = (1.0 - pow(e, (-(q * 0.016 * time) / Q / 60)))
    return P


def risk_rate(P, time, qc):
    R = 100 * (1 - pow(e, (-P * time * qc / 60)))
    return R


# Variáveis de dados
videoconfig = []
risk = [0, 0]
total_capacity = 20

# Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6") disponível na pasta /wheight
model = torch.hub.load('ultralytics/yolov5', 'yolov5s6', force_reload=True)
model.conf = 0.3
model.classes = 0
try:
    model.cuda()
except:
    model.cpu()


def videofeed(url):
    streams = streamlink.streams(url)
    feed = streams["best"].url
    return feed



# Detecção de vídeo
class VideoCamera:
    def __init__(self, source):
        # Escolhe a melhor qualidade de vídeo
        self.video = cv.VideoCapture(source)
        w = int(self.video.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT))
        #frame de detecção
        self.detection = Label(layout)
        self.detection.grid(row=0, column=0, rowspan = 2, pady=30, padx=30)
        self.bStop = Button(layout, text="STOP", command= self.__del__, font="helvetica 14")
        self.bStop.grid(row=3, column=0, pady=30)

        # Contadores de frames
        self.tinit = time.time()
        self.prev_frame_time = 0
        self.frame = 0
        self.df = {"count": [0,0],
                    "bboxes":   [[w,0],
                                [h,0],
                                [w,0],
                                [h,0]],
                    "timer": [0,0],
                    "label": [None,None]
                    }
        
        self.get_frame()

    def __del__(self):
        self.video.release()
        self.detection.destroy()
        self.bStop.destroy()
        info1.grid(row=1, column=0, pady=30, padx=30)
        url.grid(row=2, column=0, pady=30, padx=30)
        info2.grid(row=3, column=0, pady=30, padx=30)
        bDetect.grid(row=4, column=0, pady=30, padx=30)

    def get_frame(self):
        self.mask = 0
        ok, image = self.video.read()
        detect = model(image)
        detect = detect.pandas().xyxy[0]
        detect = detect.to_numpy()
        for people in detect:
            xmin, ymin, xmax, ymax, confidence, label = int(people[0]), int(people[1]), int(people[2]), int(people[3]), \
                                                        people[4], people[6]
            cv.putText(image, str(float("{0:.2f}".format(confidence))), (xmax + 20, ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0), 1, cv.LINE_AA)
            cv.putText(image, label, (xmax + 20, ymin + 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)
            cv.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            self.df["bboxes"][0].append(xmin)
            self.df["bboxes"][1].append(ymin)
            self.df["bboxes"][2].append(xmax)
            self.df["bboxes"][3].append(ymax)
            self.df["label"].append(label)

        # Volume de um escritório padrão
        volume = 9.29 * 2.70 * total_capacity
        # volume = local_height*local_width*local_height

        # Imprime o contador pessoas detectadas por frame
        cv.putText(image, str(len(detect)) + " Pessoas", (100, 80),
                   cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
        # quantos self. por frame

        new_frame_time = time.time()
        fps = 1 / (new_frame_time - self.prev_frame_time)
        self.prev_frame_time = new_frame_time
        fps = int(fps)
        fps = str(fps)
        cv.putText(image, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)

        self.frame += 1
        if self.frame == 30:
            self.frame = 0
            self.df["timer"].append(str(timedelta(seconds=int(time.time() - self.tinit))))
            self.df["count"].append(len(detect))

            if len(detect) != 0:
                Q = len(detect) * air_flow_rate(439)
                q = 2.3666 * (0.4 + 0.6 * (len(detect) - self.mask) / (len(detect)))
                qc = quanta_concentration(q, Q, ((int(time.time() - self.tinit) / 3600)), volume)
                P = infection_prob(q, Q, ((int(time.time() - self.tinit) / 3600)))
                R = risk_rate(P, ((int(time.time() - self.tinit) / 3600)), qc)
                risk.append(R)
        
        
        image = cv.resize(image,(600,450))
        img = Image.fromarray(cv.cvtColor(image,cv.COLOR_BGR2RGB))
        # Convert image to PhotoImage
        imgtk = ImageTk.PhotoImage(image = img)
        self.detection.imgtk = imgtk
        self.detection.configure(image=imgtk)
        # Repeat after an interval to capture continiously
        self.detection.after(10, self.get_frame)

def app():
    if url.get() == '':
        info1.grid(row=1, column=0)
        url.grid(row=1, column=0)
        info2.grid(row=1, column=0)
        bDetect.grid(row=1, column=0)
        return VideoCamera(0)
    else:
        info1.grid(row=1, column=0)
        url.grid(row=1, column=0)
        info2.grid(row=1, column=0)
        bDetect.grid(row=1, column=0)
        return VideoCamera(videofeed(url.get()))


layout = Tk()
layout.geometry('1280x800')
layout.eval('tk::PlaceWindow . center')
layout.title("CREM - LPO")
layout["background"] = "#423C3C"

info1 = Label(layout, text= "Para realizar a detecção em link externo, preencha o campo abaixo e clique em DETECT!", font="helvetica 14")
info1.grid(row=1, column=0, sticky=S, pady=30, padx=30)
url = Entry(layout)
url.grid(row=2, column=0, sticky=S, pady=30, padx=30)
info2 = Label(layout, text= "Caso queira utilizar a sua própria câmera, basta clicar em DETECT!", font="helvetica 14")
info2.grid(row=3, column=0, sticky=S, pady=30, padx=30)

bDetect = Button(layout, text="DETECT", command= app)
bDetect.grid(row=4, column=0, sticky=S, pady=30, padx=30)



layout.mainloop()