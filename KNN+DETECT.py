import cv2 as cv
import numpy as np
import time

kernel = None

video = cv.VideoCapture('video/walking.avi')
fgmask = cv.createBackgroundSubtractorKNN(detectShadows= True, dist2Threshold= 0)

while True:
    ok, frame = video.read()
    if frame is None:
        break
    background = fgmask.apply(frame)
    stop = cv.waitKey(30)
    if stop == 'q' or stop == 27:
        break
    if stop == 'q' or stop == 32:
        time.sleep(1)

    real = cv.bitwise_and(frame, frame, mask= background)
    background_2 = cv.cvtColor(background, cv.COLOR_GRAY2BGR)

    _, background = cv.threshold(background, 250, 255, cv.THRESH_BINARY)
    background = cv.erode(background, kernel, iterations=1)
    background = cv.dilate(background, kernel, iterations=2)

    contorno, _ = cv.findContours(background, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    frameCopia = frame.copy()
    count = 0

    for cont in contorno:
        if cv.contourArea(cont)>250:
            x, y, l, a = cv.boundingRect(cont)

            cv.rectangle(frameCopia, (x, y), (x+l, y+a), (0,0,255), 2)
            cv.putText(frameCopia, 'Pessoa',(x,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0,255,0), 1, cv.LINE_AA)

            cv.rectangle(real, (x, y), (x + l, y + a), (0, 0, 255), 2)
            cv.putText(real, 'Pessoa', (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1, cv.LINE_AA)
            count+=1

    cv.putText(real, str(count) + " Pessoas", (100, 80),
               cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
    cv.putText(frameCopia, str(count) + " Pessoas", (100, 80),
               cv.FONT_HERSHEY_SIMPLEX, .75, (8, 0, 255), 2)
    stacked = np.hstack((real, frameCopia))
    cv.imshow("Juntos", cv.resize(stacked,None,fx=1.2,fy=1.2))

video.release()
cv.destroyAllWindows()