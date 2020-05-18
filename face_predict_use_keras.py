# _*_ coding:utf-8 _*_

import cv2
import sys
import gc
from face_train_use_keras import Model

if __name__ =='__main__':

    #if len(sys.argv)!=2:
        #print('Usage:%s camera_id\r\n' % (sys.argv[0]))
        #sys.exit(0)

    #加载模型
    model=Model()
    model.load_model(file_path='./model/me.face.model.h5')

    #矩形边框的颜色
    color=(0,255,0)

    #捕捉指定摄像头的实时视频流
    cap=cv2.VideoCapture(0)

    #面部识别的分类器
    cascade_path="./haarcascades/haarcascade_frontalface_alt2.xml"

    while cap.isOpened():
        ok,frame =cap.read() #读取一帧
        if not ok:
            break

        #图像灰度化，降低计算复杂度
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        #使用人脸识别分类器，读入分类器
        cascade =cv2.CascadeClassifier(cascade_path)

        #利用分类器识别出哪个区域为人脸
        faceRects=cascade.detectMultiScale(frame_gray,scaleFactor=1.2,
                                          minNeighbors=3,minSize=(32,32))
        if len(faceRects)>0:
            print("num:",len(faceRects))
            for faceRect in faceRects:
                x,y,w,h=faceRect
                #截取脸部图像提交给模型识别这是谁
                image=frame[y-10:y+h+10,x-10:x+w+10]
                faceID=model.face_predict(image)

                if faceID==0:
                    cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),color,
                                  thickness=2)

                    #文字提示是谁
                    cv2.putText(frame, "Sun",(x+30,y+30),cv2.FONT_HERSHEY_SIMPLEX,
                                1,(255,0,255),2)
                else:
                    pass
        cv2.imshow("Face recognition:", frame)

        #等待10毫秒看是否有按键输入
        k=cv2.waitKey(10)
        #如果输入q退出循环
        if k & 0xFF == ord('q'):
            break

    #free摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
        










        
    
