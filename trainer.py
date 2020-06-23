import os
import cv2
import numpy as np
from PIL import Image as im

recognizer = cv2.face.LBPHFaceRecognizer_create()
path="dataset"

def getimages(path):
    imagepath=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    id=[]
    for ip in imagepath:
        faceimg=im.open(ip).convert("L")
        facenp=np.array(faceimg,"uint8")
        fid=int(os.path.split(ip)[-1].split(".")[1])
        faces.append(facenp)
        id.append(fid)
        cv2.waitKey(5)
    return np.array(id),faces

id,faces=getimages(path)
recognizer.train(faces,id)
recognizer.save("Recognizer/trainingData.yml")
cv2.destroyAllWindows()