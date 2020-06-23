import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap=cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer/trainingData.yml")
font=cv2.FONT_HERSHEY_COMPLEX
id=0;
cap.set(3,1000)
cap.set(4,800)

while (cap.isOpened()):
    ret, img = cap.read()
    img = img.astype('uint8')
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        id,conf=rec.predict(gray[y:y+h,x:x+w])
        if conf>70:
            pass
            tt=str(id)
        else:
            id="unknown"
        print(conf)
        img=cv2.putText(img ,str(id),(x,y-10),font,1,(255,255,0),1,cv2.LINE_AA)
    cv2.imshow("face",img)
    if cv2.waitKey(1)==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()