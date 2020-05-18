import os
import sys
import numpy as np
import cv2

IMAGE_SIZE=64

#按照指定图像大小调整尺寸
def resize_image(image,height=IMAGE_SIZE,width=IMAGE_SIZE):
    top,bottom,left,right=(0,0,0,0)
    #获得图片的尺寸
    h,w,_=image.shape

    #对应长宽不等的图片，找到最长的一边
    longest_edge=max(h,w)

    #计算短边需要增加多少像素宽度使其与长边相等
    if h<longest_edge:
        dh=longest_edge -h
        top=dh//2
        bottom=dh-top
    elif w<longest_edge:
        dw=longest_edge
        left=dw//2
        right=dw-left
    else:
        pass

    #rgb颜色
    BLACK=[0,0,0]
    #给图像增加边长，使得图片长宽相等，cv2.BORDER_CONSTANT指定边界颜色由value指定
    constant=cv2.copyMakeBorder(image,top,bottom,left,right,
                                cv2.BORDER_CONSTANT,value=BLACK)

    #返回调整好的图像
    return cv2.resize(constant,(height,width))

#取得训练数据
images=[]
labels=[]
def read_path(path_name):
    for dir_item in os.listdir(path_name):
        #获得绝对路径
        #print(dir_item)
        full_path=os.path.abspath(os.path.join(path_name,dir_item))

        #print(full_path)

        #如果是文件夹，继续递归
        if os.path.isdir(full_path):
            read_path(full_path)
        else:
            if dir_item.endswith('.jpg'):
                image=cv2.imread(full_path)
                image=resize_image(image,IMAGE_SIZE,IMAGE_SIZE)

                images.append(image)
                labels.append(path_name)
    return images,labels

#从指定路径读取训练数据
def load_dataset(path_name):
    images,labels=read_path(path_name)
    #将输入的所有图片转成四维数组，尺寸为数量*IMAGE_SIZE*IMAGE_SIZE*3
    #100张图片，图片为64*64像素，一个像素3个颜色值
    images=np.array(images)
    print(images.shape)

    #标注数据
    labels=np.array([0 if label.endswith('me') else 1  for label in labels])
    return images,labels

if __name__ =='__main__':
    #if len(sys.argv)!=2:
        #print("Usage:%s path_name\r\n"%(sys.argv[0]))
    #else:
        images,labels=load_dataset('./data')
