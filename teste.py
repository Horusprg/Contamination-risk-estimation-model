#IMPORTAMOS AS BIBLIOTECAS
import cv2 as cv
from sqlalchemy import null
import torch
import pandas
import streamlink
import time
#CARREGAMOS O MODELO TREINADO
model = torch.hub.load('yolov5', 'custom', path='wheight/YoloV5s6.pt', source='local', device="cpu")
model.conf = 0.3
model.iou = 0.10

#URL do vídeo de stream
url = "https://youtu.be/U2qwkqgLYAw"
streams = streamlink.streams(url)

#capture = cv.VideoCapture(streams["best"].url)
capture = cv.VideoCapture(0)
peoples = []

prev_frame_time = 0
 
# used to record the time at which we processed current frame
new_frame_time = 0

while True:
    ret, frame = capture.read()

    detect = model(frame)
    peoples = detect.pandas().xyxy[0]
    peoples = peoples.to_numpy()
    for people in peoples:
        xmin, ymin, xmax, ymax, confidence, label = int(people[0]), int(people[1]), int(people[2]), int(people[3]), people[4], people[6]
        cv.rectangle(frame, (xmin, ymin), (xmax, ymax), (255,255,255), 2)
        cv.putText(frame, str(confidence),(xmax + 20,ymin), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
        cv.putText(frame, str(label),(xmax + 20,ymin+40), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1, cv.LINE_AA)
    posicoes = []
    
    new_frame_time = time.time()
    
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
 
    # converting the fps into integer
    fps = int(fps)
 
    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)
 
    # putting the FPS count on the frame
    cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv.LINE_AA)

    print(fps)
    cv.imshow('Frame', frame)
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break