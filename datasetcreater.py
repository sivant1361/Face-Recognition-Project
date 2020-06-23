import cv2
import numpy as np
face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# import mysql.connector

# def insertdat(id,name):
#     mydb = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         passwd="9994532266",
#         database="facedet"
#     )
#
#     conn = mydb.cursor()
#     conn.execute("use facedet")
#     conn.execute("create table dbase(id int,name varchar(20)")
#     cur=conn.execute("SELECT * FROM dbase where id="+str(id))
#     print(cur)
#     myresult = conn.fetchall()
#     isrecordexist=0
#     for row in myresult:
#         isrecordexist=1
#     if(isrecordexist==1):
#         cm="update fbase set name="+str(name)+" where id="+str(id)
#     else:
#         cm="insert into fbase set values("+str(id)+","+str(name)+")"
#     conn.execute(cm)
#     conn.commit()
#     conn.close()

cap=cv2.VideoCapture(0)
id=int(input("Enter your ID:"))
name=input("Enter your name:")
# insertdat(id,name)
samplenum=0;
while (cap.isOpened()):
    ret, img = cap.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        samplenum+=1
        cv2.imwrite('dataset/user.'+str(id)+"."+str(samplenum)+".jpg",gray[y:y+h,x:x+w])
        cv2.waitKey(100);
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.imshow("face",img)
    cv2.waitKey(1)
    if (samplenum==80):
        break

cap.release()
cv2.destroyAllWindows()
