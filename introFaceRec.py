import cv2
import numpy as np

faceDetect=cv2.CascadeClassifier(r'C:/FR_Nimoy/haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0);
while(True):
    ret,img=cam.read();
    gray = cv2.imread(img,0)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow("Myface", img);
        if(cv2.waitKey(1)==ord('q')):
            break;
        cam.release()
        cv2.detroyAllWindows()
