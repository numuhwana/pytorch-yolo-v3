from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse
from PIL import Image
from numba import vectorize
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import lanedet as ld 
import testclassify as tff #trafficlight를 위함
from skimage.transform import resize
import cnn_sign as tfs 
import os

def laneDetection(frame):
    #ret, frame = cap.read()
    if frame.shape[0] != 540:  # resizing for challenge video
        frame = cv2.resize(frame, None, fx=3 / 4, fy=3 / 4, interpolation=cv2.INTER_AREA)
        result = ld.detect_lanes_img(frame)
    return result

#c1[0]: left c2[0]: right c1[1]:top c2[1]:bottom
#빨간불=0 노란불=1 초록불=2 범위밖=3
def getTraffic(c1,c2,img,tfmod,Gf,Rf):
    #이미지 형태로 바꾸기
    crop_img=img[0:500,400:1500]
    if 500<int(c2[1]) or int(c1[0])<400 or 1500<int(c2[0]):
        return img, False
    crop_img=img[int(c1[1]):int(c2[1]),int(c1[0]):int(c2[0])]#자르기
    crop_img = cv2.resize(crop_img, (64, 64))#크기 수정
    crop_img = crop_img[...,::-1].astype(np.float32)#RGB->BGR
    predpic=tff.classify_traffic(crop_img,tfmod,Gf,Rf)
    predpic=Image.fromarray(predpic)
    img= Image.fromarray(img)
    img.paste(predpic,(50,50))
    img=np.array(img)
    return img, True #여기서 신호등 붙힌거 리턴

def SignDet(c1,c2,img,signmodl,Cu,In,Rg):
    plane=(int(c2[0])-int(c1[0]))*(int(c2[1])-int(c1[1]))
    if plane<42600:
        crop_img=img[int(c1[1]):int(c2[1]),int(c1[0]):int(c2[0])]#자르기
        crop_img = cv2.resize(crop_img, (28, 28))#크기 수정
        crop_img = crop_img[...,::-1].astype(np.float32)#RGB->BGR
        predpic=tfs.predictsign(signmodl,crop_img,Cu,In,Rg)
        predpic=Image.fromarray(predpic)
        img= Image.fromarray(img)
        if (int(c1[1])-100)>=0:
            img.paste(predpic,(int(c1[0]),int(c1[1])-100))
        else:
            img.paste(predpic,(int(c1[0]),int(c1[1])+100))
        img=np.array(img)
        return img,True
    else:
        return img,False


# 변수이름: img_:변환한 이미지, orig_im: 원래 이미지, dim: 각 이미지 사이즈(가로,세로) ex: (1920, 1080)
def prep_image(img, inp_dim):#이미지변환
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim))) #이미지를 정보화
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)#이미지를 숫자로 표현
    return img_, orig_im, dim #img_ is divided!! -> value b2n 0 and 1

def write(x, img,tfmod,Gf,Rf,signmodl,Cu,In,Rg):
    #print("x: ",x)
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls]) #클래스에서 이름 골라서 넣기
    #원하는 클래스만 출력
    thic=3
    ign=True
    if label=="car" or label=="traffic light" or label=="person" or label=="truck" or label=="train" or label=="bus" or label=="bicycle" or label=="motorbike" or label=="stop sign":
        if label=="car":#차: 빨간색
            color = [34,34,178]
        elif label=="traffic light":#신호등: 검정색
            color = [255,255,255]
            thic=1
            img,ign=getTraffic(c1,c2,img,tfmod,Gf,Rf)
        elif label=="person":#사람: 하얀색
            color = [255,255,255]
        elif label=="truck":
            color = [42,42,165] #트럭: 갈색
        elif label=="train": #기차:파란색
            color = [178,34,34]
        elif label=="bus": #버스: 빨간색
            color = [30,105,210]
        elif label=="bicycle": #자전거: 초록색
            color = [34,178,34]
        elif label=="motorbike": #오토바이: 초록색
            color = [34,178,34]
        elif label=="stop sign": #스톱사인: 노란색
            color = [34,178,178]
            thic=1
            img,ign=SignDet(c1,c2,img,signmodl,Cu,In,Rg)
        #c1[0]: left c2[0]: right c1[1]:top c2[1]:bottom
        if int(c1[0])<=952 and int(c2[0])>=952 and int(c1[1])<=900 and c2[1]>=900: #내차는 출력하지 않기 위함
            return img
        elif ign==True:
            cv2.rectangle(img, c1, c2, color, thic)
            if label != "traffic light" and label!="stop sign":
                t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
                c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
                cv2.rectangle(img, c1, c2,color, -1)
                cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img

def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest = 'video', help = 
                        "Video to run detection upon",
                        default = "video/choyi_test2.mp4", type = str)
    parser.add_argument("--dataset", dest = "dataset", help = "Dataset on which the network has been trained", default = "pascal")
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.45)
    parser.add_argument("--cfg", dest = 'cfgfile', help = 
                        "Config file",
                        default = "cfg/yolov3_yujin.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = 
                        "weightsfile",
                        default = "weight/yolov3-custom_510000_yujin.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "416", type = str)
    return parser.parse_args()

# 변수명: model: 디텍션시 필요한 neural 네트워크 CUDA: GPU 사용을 위한 변수  

if __name__ == '__main__':

    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
    tfmod,Gf,Rf=tff.load_net()  # loaded_model from testclassify.py
    signmodl,Cu,In,Rg=tfs.loadsignmodel() #load from sign

    CUDA = torch.cuda.is_available()

    num_classes = 80
    
    bbox_attrs = 5 + num_classes
    
    print("Loading network.....")
    model = Darknet(args.cfgfile) #다크네트의 네트워크 가져오기
    model.load_weights(args.weightsfile) #weight값 넣기
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32

    if CUDA:
        model.cuda()
        
    #model(get_test_input(inp_dim, CUDA), CUDA) #테스트부분

    model.eval()
    
    videofile = args.video
    print(videofile)
    cap = cv2.VideoCapture(videofile)
    
    assert cap.isOpened(), 'Cannot capture source'
        
    frames = 0
    start = time.time()    
    while cap.isOpened():
            
        ret, frame = cap.read()# frame으로 가져오는 부분
        if ret:
                
            frame=laneDetection(frame)
            img, orig_im, dim = prep_image(frame, inp_dim) #이미지 정보에 대한것 가져오기
            im_dim = torch.FloatTensor(dim).repeat(1,2)                        
                
                
            if CUDA:
                im_dim = im_dim.cuda()
                img = img.cuda()
                
            with torch.no_grad():   
                output = model(Variable(img), CUDA)
            output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
            if type(output) == int:
                frames += 1
                #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
                cv2.imshow("frame", orig_im)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q'):
                    break
                continue
               
            im_dim = im_dim.repeat(output.size(0), 1)
            scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
                
            output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
            output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
                
            output[:,1:5] /= scaling_factor
        
            for i in range(output.shape[0]): #객체
                output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
                output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
                
            classes = load_classes('data/coco.names')
            colors = pkl.load(open("pallete", "rb"))


            for i in range(len(output)):
                orig_im=write(output[i],orig_im,tfmod,Gf,Rf,signmodl,Cu,In,Rg)
            #list(map(lambda x: write(x, orig_im,tfmod,Gf,Rf), output))
            cv2.UMat(cv2.imshow("frame", orig_im))
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            #print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))

                
        else:
            break
    

    
    
