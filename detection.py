# Imports do projeto
import cv2 as cv
from math import e
import time
import streamlink
import torch
from datetime import timedelta
import csv

# Stream da detecção de vídeo
def gen(camera):
    while True:
        frame = camera.get_frame(df)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


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

def csv_save(test):
    with open("test.csv", "w") as outfile:
   
        # pass the csv file to csv.writer.
        writer = csv.writer(outfile)
        
        # convert the dictionary keys to a list
        key_list = list(test.keys())
        
        # find the length of the key_list
        limit = len(key_list)
        
        # the length of the keys corresponds to
        # no. of. columns.
        writer.writerow(test.keys())
        
        # iterate each column and assign the
        # corresponding values to the column
        for i in range(limit):
            writer.writerow([test[x][i] for x in key_list])


# Variáveis de dados
videoconfig = []
risk = [0, 0]

labels = {"with_mask": 0, "without_mask": 0, "mask_weared_incorrect": 0}
df = {"count": [],
      "bboxes": [[], [], [], []],
      "timer": [],
      "label": [],
      "risk": []
      }

# Carregando o modelo do yolov5("YoloV5s", "YoloV5m", "YoloV5l", "YoloV5xl", "YoloV5s6") disponível na pasta /wheight
model = torch.hub.load('yolov5', 'custom', path='wheight/yoloV5n6.pt', source='local')
model.conf = 0.5
model.iou = 0.5


# Detecção de vídeo
class VideoCamera(object):
    def __init__(self, qgrate, area):
        # Escolhe a melhor qualidade de vídeo
        self.video = cv.VideoCapture(0)
        self.qgrate = qgrate
        self.area = area
        # Contadores de frames
        self.tinit = time.time()
        self.prev_frame_time = 0
        self.mask = 0
        self.frame = 0
        df["bboxes"][0].append(int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)))
        df["bboxes"][1].append(int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        df["bboxes"][2].append(int(self.video.get(cv.CAP_PROP_FRAME_WIDTH)))
        df["bboxes"][3].append(int(self.video.get(cv.CAP_PROP_FRAME_HEIGHT)))
        df["count"].append(0)
        df["label"].append(None)
        df["timer"].append(0)
        df["bboxes"][0].append(0)
        df["bboxes"][1].append(0)
        df["bboxes"][2].append(0)
        df["bboxes"][3].append(0)
        df["count"].append(0)
        df["label"].append(None)
        df["timer"].append(0)
        df["risk"].append(0)
        df["risk"].append(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, df):
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
            df["bboxes"][0].append(xmin)
            df["bboxes"][1].append(ymin)
            df["bboxes"][2].append(xmax)
            df["bboxes"][3].append(ymax)
            df["label"].append(label)
            labels[label] += 1
            if label == "with_mask":
                self.mask += 1

        # Volume de um escritório padrão
        volume = int(self.area) * 3
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
            df["timer"].append(str(timedelta(seconds=int(time.time() - self.tinit))))
            df["count"].append(len(detect))

            if len(detect) != 0:
                Q = len(detect) * air_flow_rate(439)
                q = float(self.qgrate) * (0.4 + 0.6 * (len(detect) - self.mask) / (len(detect)))
                qc = quanta_concentration(q, Q, ((int(time.time() - self.tinit) / 3600)), volume)
                P = infection_prob(q, Q, ((int(time.time() - self.tinit) / 3600)))
                R = risk_rate(P, ((int(time.time() - self.tinit) / 3600)), qc)
                risk.append(R)
                df["risk"].append(R)
        
        ok, jpeg = cv.imencode('.jpg', image)
        return jpeg.tobytes()
