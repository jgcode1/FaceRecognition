# -*- coding: cp936 -*-  

import cv2
import numpy as np

def RecogFaceByPic(pic_path):
    img=cv2.imread(pic_path)#载入一张包含人脸的图片
    
    cascade_path='./haarcascades/haarcascade_frontalface_alt2.xml'

    cas=cv2.CascadeClassifier(cascade_path)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #检测人脸：跟数据库进行对比
    #结果：人脸的坐标x,y,长度，宽度
    rects=cas.detectMultiScale(gray)

    for x,y,width,height in rects:
        cv2.rectangle(img,(x,y),(x+width,y+height),(0,0,255),2)

    cv2.imshow('face',img)

    #k=cv2.waitKey(10)
    #if k & 0xFF == ord('q'):
     #   break

    #img.release()
    #cv2.destoryAllWindows()

    
#检测人脸函数
def repeat():
    cv2.namedWindow('W1')
    capture =cv2.VideoCapture(0)
    
    while capture.isOpened():
    #frame =cv2.queryFrame(capture)#每次从摄像头获取一张图片
        ok,frame=capture.read()
        if not ok:
            break
        
        grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)#转换为灰度图片
        #storage=cv2.CreateMemStorage(0)#创建一个内存空间，人脸检测用

        #cv2.EqualizeHist(grey,grey)#将灰度图像直方图均衡化，可以使得灰度
    #图像信息量减少，加快检测速度
        classfier=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
        classfier.load('D:\python\haarcascades\haarcascade_frontalface_alt2.xml')

    #检测图片中的脸，返回一个包含了人脸信息的对象faces
        faces=classfier.detectMultiScale(grey,scaleFactor=1.2,minNeighbors=3,minSize=(32,32))
    #获得人脸所在位置的数据
        for x,y,w,h in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.imshow('W1',frame)

        k=cv2.waitKey(10)
        if k & 0xFF ==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    repeat()    
    #RecogFaceByPic('./data/me/1.jpg')
