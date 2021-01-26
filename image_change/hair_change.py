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


def hair_image(event,userid,color):
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
    #hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>5) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] + 150,hsv[:, :, 0])
    #hsv_2[:, :, 2] = np.where((hsv[:, :, 2]>5) )
    #hsv_2[:, :, 2] = np.where((hsv[:, :, 0]>5) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] *0.5,hsv[:, :, 2])#金髪から黒髪
    #hsv_2[:, :, 2] = np.where((hsv[:, :, 0]>5) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] *0.5,hsv[:, :, 2])
    #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] +110 ,hsv[:, :, 0])#黒(茶色)から青
    #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] +50 ,hsv[:, :, 0])#黒(茶色)から緑
    #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] +160 ,hsv[:, :, 0])#黒(茶色)から赤
    #bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
    #cv2.imwrite(output_path,bgr)
    #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] +50 ,hsv[:, :, 0])
    if color == 1:
        #色相を０に
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>=0) & (hsv[:, :, 2]<30) ,hsv[:, :, 0] *0.001 ,hsv[:, :, 0])
        #hsv_3[:, :, 0] = np.where((hsv_2[:, :, 2]>30) & (hsv_2[:, :, 2]<60) ,hsv_2[:, :, 0] *0.001 ,hsv_2[:, :, 0])#150~180
        #hsv_4[:, :, 0] = np.where((hsv_3[:, :, 2]>60) & (hsv_3[:, :, 2]<90) ,hsv_3[:, :, 0] *0.001 ,hsv_3[:, :, 0])#160~190
        #明度をあげる
        ##hsv_5[:, :, 2] = np.where((hsv_4[:, :, 2]>=0) & (hsv_4[:, :, 2]<30) ,hsv_4[:, :, 2] + 140 ,hsv_4[:, :, 2])#140~170
        ##hsv_6[:, :, 2] = np.where((hsv_5[:, :, 2]>30) & (hsv_5[:, :, 2]<60) ,hsv_5[:, :, 2] + 120 ,hsv_5[:, :, 2])#150~180
        ##hsv_7[:, :, 2] = np.where((hsv_6[:, :, 2]>60) & (hsv_6[:, :, 2]<90) ,hsv_6[:, :, 2] + 100 ,hsv_6[:, :, 2])#160~190
#ここから
        #彩度を上げる
        hsv_5[:, :, 1] = np.where((hsv_4[:, :, 2]>=0) & (hsv_4[:, :, 2]<30) ,hsv_4[:, :, 1] + 230 ,hsv_4[:, :, 1])
        #hsv_6[:, :, 1] = np.where((hsv_5[:, :, 2]>30) & (hsv_5[:, :, 2]<60) ,hsv_5[:, :, 1] + 230 ,hsv_5[:, :, 1])
        #hsv_7[:, :, 1] = np.where((hsv_6[:, :, 2]>60) & (hsv_6[:, :, 2]<100) ,hsv_6[:, :, 1] + 230 ,hsv_6[:, :, 1])

        


        #色相を上げる
        hsv_8[:, :, 0] = np.where((hsv_5[:, :, 2]>=0) & (hsv_5[:, :, 2]<30) ,hsv_5[:, :, 0] +80 ,hsv_5[:, :, 0])
        #hsv_9[:, :, 0] = np.where((hsv_8[:, :, 2]>30) & (hsv_8[:, :, 2]<60) ,hsv_8[:, :, 0] +80 ,hsv_8[:, :, 0])#150~180
        #hsv_10[:, :, 0] = np.where((hsv_9[:, :, 2]>60) & (hsv_9[:, :, 2]<100) ,hsv_9[:, :, 0]+80 ,hsv_9[:, :, 0])#160~190


        #明度を上げる
        hsv_11 = np.copy(hsv_8)
        hsv_11[:, :, 2] = np.where((hsv_8[:, :, 2]>=0) & (hsv_8[:, :, 2]<30) ,hsv_8[:, :, 2] + 180 ,hsv_8[:, :, 2])#140~170
        
        hsv_12 = np.copy(hsv_11)
        #hsv_12[:, :, 2] = np.where((hsv_11[:, :, 2]>30) & (hsv_11[:, :, 2]<60) ,hsv_11[:, :, 2] + 160 ,hsv_11[:, :, 2])#150~180

        hsv_13 = np.copy(hsv_12)
        #hsv_13[:, :, 2] = np.where((hsv_12[:, :, 2]>60) & (hsv_12[:, :, 2]<100) ,hsv_12[:, :, 2] + 140 ,hsv_12[:, :, 2])#160~190
#ここまで
        #色相追加
        #彩度をあげる
        ##hsv_8[:, :, 1] = np.where((hsv_7[:, :, 2]>140) & (hsv_7[:, :, 2]<190) & (hsv_4[:, :, 2]>=0) & (hsv_4[:, :, 2]<30) ,hsv_7[:, :, 1] + 100 ,hsv_7[:, :, 1])
        #hsv_8[:, :, 1] = np.where((hsv_7[:, :, 2]>140) & (hsv_7[:, :, 2]<190) & (hsv_7[:, :, 1]<100) ,hsv_7[:, :, 1] + 100 ,hsv_7[:, :, 1])#& (hsv_4[:, :, 1]<20) 
        #hsv_9[:, :, 1] = np.where(((hsv_8[:, :, 2]>150) & (hsv_8[:, :, 2]<180)) & (hsv_8[:, :, 1]<100) ,hsv_8[:, :, 1] + 100,hsv_8[:, :, 1])
        #hsv_10[:, :, 1] = np.where(((hsv_9[:, :, 2]>160) & (hsv_9[:, :, 2]<190)) & (hsv_9[:, :, 1]<100) ,hsv_9[:, :, 1] + 100 ,hsv_9[:, :, 1])
        #hsv_5[:, :, 2] = np.where((hsv_4[:, :, 2]>90) & (hsv_4[:, :, 2]<120) ,hsv_4[:, :, 2] + 80 ,hsv_4[:, :, 2])
        #hsv_3[:, :, 1] = np.where(((hsv[:, :, 2]>30) & (hsv[:, :, 2]<128)) & (hsv[:, :, 1]<100) ,hsv_2[:, :, 1] + 100 ,hsv_2[:, :, 1],)
        #hsv_3[:, :, 2] = np.where(((hsv_2[:, :, 2]>0) & (hsv_2[:, :, 2]<200)) & (hsv_2[:, :, 1]<255) ,hsv_2[:, :, 1] -190 ,hsv_2[:, :, 2])
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] + 50,hsv[:, :, 0]) #緑
    elif color == 2:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 110,hsv[:, :, 0]) #青
    elif color == 3:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 20,hsv[:, :, 0]) #黄色
    elif color == 4:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 176,hsv[:, :, 0]) #ピンク
        hsv_2[:, :, 1] = np.where((hsv[:, :, 0]>=176) ,hsv[:, :, 1] + 96 ,hsv[:, :, 1])
    elif color == 5:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤色
    elif color == 6:
        hsv_2[:, :, 2] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] *0.4,hsv[:, :, 2]) #黒色




    #茶色から指定色-----------------------------------------------------------------------------------------------------------
    #if color == 1:
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<255) ,hsv[:, :, 0] +50 ,hsv[:, :, 0])#黒(茶色)から緑
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] + 50,hsv[:, :, 0]) #緑
    #elif color == 2:
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 110,hsv[:, :, 0]) #青
    #elif color == 3:
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 20,hsv[:, :, 0]) #黄色
    #elif color == 4:
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 176,hsv[:, :, 0]) #ピンク
        #hsv_2[:, :, 1] = np.where((hsv[:, :, 0]>=176) ,hsv[:, :, 1] + 96 ,hsv[:, :, 1])
    #elif color == 5:
        #hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>6) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤色
    #elif color == 6:
        #hsv_2[:, :, 2] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] *0.4,hsv[:, :, 2]) #黒色
    #hsv_2[:, :, 2] = np.where((hsv_2[:, :, 0]>6) & (hsv_2[:, :, 0]<30) ,hsv_2[:, :, 1] *0.7,hsv_2[:, :, 2]) #黒色
    #0.001 赤
    #0.3 緑
    #bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
    #----------------------------------------------------------------------------------------------------------------------

    bgr = cv2.cvtColor(hsv_11, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, bgr)

#hsv_2[:, :, 0] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 0] + 176,hsv[:, :, 0]) #濃い茶色
#hsv_2[:, :, 1] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 1] + 96 ,hsv[:, :, 1])


#白
#hsv_2[:, :, 1] = np.where((hsv[:, :, 2]>0) & (hsv[:, :, 2]<128) ,hsv[:, :, 1] + 180 ,hsv[:, :, 1])
#hsv_3[:, :, 2] = np.where((hsv_2[:, :, 2]>0) & (hsv_2[:, :, 2]<128) ,hsv_2[:, :, 2] + 180 ,hsv_2[:, :, 2])