import cv2
import numpy as np 



face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')


cap = cv2.VideoCapture(0)

i = 0

while(30):
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)
    
    for (x,y,w,h) in faces: 
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        if i % 5 == 0:  
            img_item = str(i) + ".png"
            cv2.imwrite(img_item, roi_color)
        i+=1
        color = (255,0,0)
        stroke = 2
        end_cord_x = x+w
        end_cord_y = y+h
        cv2.rectangle(frame, (x,y), (end_cord_x, end_cord_y), color, stroke)
        
        
        cv2.imshow('picture', frame)
        
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# When everything done, release the capture 
cap.release()
cv2.destroyAllWindows()
        
        