import cv2
import numpy as np
from cv2.data import haarcascades

face_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_frontalface_default.xml') # data/haarcascade/haarcascade_frontalface_default.xml
eye_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('data/haarcascade/haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize = (30,30))


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # roi - region of interest

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y + h, x:x + w]


        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize = (10,10))

        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

        smile = smile_cascade.detectMultiScale(roi_gray, 1.7, 5, minSize = (15, 15))

        for (qx, qy, qw, qh) in smile:
            cv2.rectangle(roi_color, (qx, qy), (qx + qw, qy + qh), (0, 0, 255), 2)


    cv2.putText(
        frame, f'Faces deteceted: {len(faces)}',
        (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1
    )

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
