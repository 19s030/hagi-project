from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage, FlexSendMessage,CarouselContainer,BubbleContainer
import main
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import path_data
import random


def hair_image2(event,userid,color):
    image_path, output_path = path_data.get_image_path(event,userid)


    img = cv2.imread(image_path)     # Load image
 

 
    #height = img.shape[0]
    #width = img.shape[1]
    #img2 = cv2.resize(img , (int(width*0.5), int(height*0.5)))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # BGR->HSV変換
    hsv_2 = np.copy(hsv)
    hsv_3 = np.copy(hsv_2)
    hsv_4 = np.copy(hsv_3)
    hsv_5 = np.copy(hsv_4)
    hsv_6 = np.copy(hsv_5)
    hsv_7 = np.copy(hsv_6)
    hsv_8 = np.copy(hsv_7)
    hsv_9 = np.copy(hsv_8)
    hsv_10 = np.copy(hsv_9)


    #茶色から指定色-----------------------------------------------------------------------------------------------------------
    if color == 1:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] +50 ,hsv[:, :, 0])#黒(茶色)から緑
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] + 50,hsv[:, :, 0]) #緑
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    elif color == 2:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 110,hsv[:, :, 0]) #青
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    elif color == 3:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 20,hsv[:, :, 0]) #黄色
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    elif color == 4:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 176,hsv[:, :, 0]) #ピンク
        hsv_3[:, :, 1] = np.where((hsv_2[:, :, 0]>=176) ,hsv_2[:, :, 1] + 96 ,hsv_2[:, :, 1])
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    elif color == 5:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤色
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    elif color == 6:
        hsv_2[:, :, 1] = np.where((hsv[:, :, 0]>=0) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] * 0,hsv[:, :, 1]) #黒色
        hsv_3[:, :, 2] = np.where((hsv_2[:, :, 0]>6) & (hsv_2[:, :, 0]<30) ,hsv_2[:, :, 0] *0,hsv_2[:, :, 0]) #黒色
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
        cv2.imwrite(output_path, bgr)
    #0.001 赤
    #0.3 緑
    #bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
    #----------------------------------------------------------------------------------------------------------------------

    

#hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 176,hsv[:, :, 0]) #濃い茶色
#hsv_2[:, :, 1] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 1] + 96 ,hsv[:, :, 1])


#白
#hsv_2[:, :, 1] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 1] + 180 ,hsv[:, :, 1])
#hsv_3[:, :, 2] = np.where((hsv_2[:, :, 2]>0) & (hsv_2[:, :, 2]<128) ,hsv_2[:, :, 2] + 180 ,hsv_2[:, :, 2])