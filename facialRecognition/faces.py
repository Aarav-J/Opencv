import numpy as np
import cv2
import pickle

recognizer = cv2.face_LBPHFaceRecognizer.create()
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
    
cap = cv2.VideoCapture(0)
i = 0
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    
    for (x,y,w,h) in faces: 
        # print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        id_, conf  = recognizer.predict(roi_gray)
        if conf>=45 and conf <= 85: 
            
            print(id_)
            name = labels[id_]
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255,255,255)
            stroke = 3
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        color = (255,0,0)
        stroke = 5
        cv2.rectangle(frame, (x,y), (x+w, y+h), color, stroke)
        i += 1
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows()
