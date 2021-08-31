import cv2 as cv
import numpy as np
import time
kernel = None
from math import sqrt

video = cv.VideoCapture('video/walking.avi')
fgmask = cv.createBackgroundSubtractorKNN(detectShadows= True, dist2Threshold= 0)

#contagem de pessoas detectadas por segundo
count = 0
#contagem de quadros por segundo
segundo = 0

while True:
    #Define o frame atual e caso seja recebido, continua o codigo
    ok, frame = video.read()
    if frame is None:
        break

    #Aplica a mascara no frame atual
    background = fgmask.apply(frame)

    #Define as keys para pausa e break do processo
    stop = cv.waitKey(30)
    if stop == 'q' or stop == 27:
        break
    if stop == 'q' or stop == 32:
        time.sleep(3)

    #Define o frame com tratamento da mascara, mas com retorno de cor
    real = cv.bitwise_and(frame, frame, mask= background)

    #mascara auxiliar para tratamento de imagem
    background_2 = cv.cvtColor(background, cv.COLOR_GRAY2BGR)
    _, background = cv.threshold(background, 250, 255, cv.THRESH_BINARY)
    background = cv.erode(background, kernel, iterations=1)
    background = cv.dilate(background, kernel, iterations=2)

    #encontra os objetos na imagem tratada
    contorno, _ = cv.findContours(background, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    frameCopia = frame.copy()
    posicoes = []

    #Realiza a busca das pessoas no frame
    for cont in contorno:
        if cv.contourArea(cont)>250:
            x, y, l, a = cv.boundingRect(cont)
            horC = int(x + l/2)
            verC = int(y + a/2-10)
            cv.putText(frameCopia, 'Pessoa',(x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(real, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv.putText(real, 'Pessoa', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)
            count+=1
            posicoes.append([horC,verC,y+a,int(l/2),int(a/2)])

    #Cria dois laços para detectar distancia entre pessoas proximas
    for j in range(len(posicoes)):
        aux = float("inf")
        for i in range(len(posicoes)):
            dist = sqrt((posicoes[i][0] - posicoes[j][0]) ** 2) + ((posicoes[i][2] - posicoes[j][2]) ** 2)
            if dist < aux and dist != 0:
                aux = dist

        #Define o circulo de avaliação de perigo
        if aux < 150:
            cv.ellipse(frameCopia,(posicoes[j][0],posicoes[j][2]),(25,5),0,0,360,(0,0,255),2)
            cv.circle(frameCopia, (posicoes[j][0], posicoes[j][1]), 7, (0, 0, 255), -1)
            cv.line(frameCopia, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0],posicoes[j][2] ), (0, 0, 255), 2)

            # Exibe o objeto a menor distancia encontrado da referencia
            cv.putText(frameCopia, str(aux), (posicoes[j][0] + 40, posicoes[j][2] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (0, 0, 255), 1, cv.LINE_AA)
        elif aux > 150 and aux < 300:
            cv.ellipse(frameCopia,(posicoes[j][0],posicoes[j][2]),(25,5),0,0,360,(0,255,255),2)
            cv.circle(frameCopia, (posicoes[j][0], posicoes[j][1]), 7, (0, 255, 255), -1)
            cv.line(frameCopia, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0], posicoes[j][2]), (0, 255, 255), 2)

            # Exibe o objeto a menor distancia encontrado da referencia
            cv.putText(frameCopia, str(aux), (posicoes[j][0] + 40, posicoes[j][2] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (0, 255, 255), 1, cv.LINE_AA)
        if aux > 300:
            cv.ellipse(frameCopia,(posicoes[j][0],posicoes[j][2]),(25,5),0,0,360,(0,255,0),2)
            cv.circle(frameCopia, (posicoes[j][0], posicoes[j][1]), 7, (0, 255, 0), -1)
            cv.line(frameCopia, (posicoes[j][0], posicoes[j][1]), (posicoes[j][0], posicoes[j][2]), (0, 255, 0), 2)

            # Exibe o objeto a menor distancia encontrado da referencia
            cv.putText(frameCopia, str(aux), (posicoes[j][0] + 40, posicoes[j][2] + 20), cv.FONT_HERSHEY_SIMPLEX, 0.3,
                       (0, 255, 0), 1, cv.LINE_AA)

    #implementa a contagem de pessoas e frames para gerar a media
    segundo +=1
    pessoas = int(count / segundo)

    #Imprime o contador de média de pessoas detectadas por segundo
    cv.putText(real, str(pessoas) + " Pessoas", (100, 80),
               cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
    cv.putText(frameCopia, str(pessoas) + " Pessoas", (100, 80),
               cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)

    #concatena as duas imagens e as exibe
    stacked = np.hstack((real, frameCopia))
    cv.imshow("Juntos", cv.resize(stacked,None,fx=1.2,fy=1.2))

    # quantos frames por segundo
    if segundo == 10:
        segundo = 0
        count = 0