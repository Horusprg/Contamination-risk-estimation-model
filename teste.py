#IMPORTAMOS AS BIBLIOTECAS
import cv2 as cv
from sqlalchemy import null
import torch
import pandas
import streamlink

#CARREGAMOS O MODELO TREINADO
model = torch.hub.load('yolov5', 'custom', path='wheight/yolov5s6-peb.pt', source='local')
model.conf = 0.3
model.iou = 0.10

#URL do vídeo de stream
url = "https://youtu.be/rkhQrCEf0xk"
streams = streamlink.streams(url)

capture = cv.VideoCapture(streams["best"].url)
peoples = []

while True:
    ret, frame = capture.read()
    	
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detect = model(gray)
    peoples = detect.pandas().xyxy[0]
    peoples = peoples.to_numpy()
    for people in peoples:
        xmin, ymin, xmax, ymax, confidence, label = int(people[0]), int(people[1]), int(people[2]), int(people[3]), people[4], people[6]
        cv.rectangle(gray, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
        cv.putText(gray, str(confidence),(xmax + 20,ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
        cv.putText(gray, str(label),(xmax + 20,ymin+40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
    posicoes = []
    
    cv.imshow('Frame', gray)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break