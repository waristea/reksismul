"""
import numpy as np
import cv2

faceCascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

    cv2.imshow('video',img)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()
"""
# https://medium.com/@madhawavidanapathirana/real-time-human-detection-in-computer-vision-part-2-c7eda27115c6
import cv2
import time
import os

person_cascade = cv2.CascadeClassifier(os.path.join('opencv/data/haarcascades/haarcascade_fullbody.xml'))
cap = cv2.VideoCapture("TownCentre.mp4")

cap.set(3,640) # set Width
cap.set(4,480) # set Height

while True:
    r, frame = cap.read()
    if r:
        start_time = time.time()
        frame = cv2.resize(frame,(640,360)) # Downscale to improve frame rate
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Haar-cascade classifier needs a grayscale image
        rects = person_cascade.detectMultiScale(gray_frame)

        end_time = time.time()
        print("Elapsed Time:",end_time-start_time)

        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),2)
        cv2.imshow("preview", frame)
    k = cv2.waitKey(1)
    if k & 0xFF == ord("q"): # Exit condition
        break
