import os
from posixpath import basename
from typing import final
from PIL import Image
import numpy as np
import cv2
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images") 

current_id = 0

label_ids = {}
x_train = []
y_labels = []

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face_LBPHFaceRecognizer.create()

for root, dirs, files in os.walk(image_dir):
    for file in files: 
        if file.endswith(".png") or file.endswith(".jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id+=1
            id_ = label_ids[label]
            # print(label_ids)
            pil_image = Image.open(path).convert("L")
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(final_image, "uint8")
            faces = face_cascade.detectMultiScale(image_array, 1.5, 5)
            
            for (x,y,w,h) in faces: 
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                
# print(y_labels)
# print(x_train)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)
    
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")