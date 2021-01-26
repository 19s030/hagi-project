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


def skin_image(event,userid,color):
    image_path, output_path = path_data.get_image_path(event,userid)
 
 
    image = cv2.imread(image_path) # Load image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) # BGR->HSV変換
    hsv_2 = np.copy(hsv)

    #a = random.randint(1,6)

    if color == 1:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤に変更
        hsv_3 = np.copy(hsv_2)
        hsv_3[:, :, 0] = np.where((hsv_2[:, :, 0]>= 0) & (hsv_2[:, :, 0]<10) ,hsv_2[:, :, 0] + 60,hsv_2[:, :, 0]) #緑
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
    elif color == 2:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤に変更
        hsv_3 = np.copy(hsv_2)
        hsv_3[:, :, 0] = np.where((hsv_2[:, :, 0]>= 0) & (hsv_2[:, :, 0]<10) ,hsv_2[:, :, 0] + 110,hsv_2[:, :, 0]) #青
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
    elif color == 3:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤に変更
        hsv_3 = np.copy(hsv_2)
        hsv_3[:, :, 0] = np.where((hsv_2[:, :, 0]>= 0) & (hsv_2[:, :, 0]<10) ,hsv_2[:, :, 0] + 25,hsv_2[:, :, 0]) #黄色
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
    elif color == 4:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] + 160,hsv[:, :, 0]) #ピンク
        hsv_3 = np.copy(hsv_2)
        hsv_3[:, :, 1] = np.where((hsv_2[:, :, 0]>140) & (hsv_2[:, :, 0]<180) ,hsv[:, :, 1] *0.8,hsv[:, :, 1]) #ピンク       
        bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
    elif color == 5:
        hsv_2[:, :, 0] = np.where((hsv[:, :, 0]>6) & (hsv[:, :, 0]<30) ,hsv[:, :, 0] *0.001,hsv[:, :, 0]) #赤色
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
    elif color == 6:
      
        hsv_2[:, :, 2] = np.where((hsv[:, :, 0]>3) & (hsv[:, :, 0]<30) ,hsv[:, :, 2] *0.4,hsv[:, :, 2]) #黒色
        bgr = cv2.cvtColor(hsv_2, cv2.COLOR_HSV2BGR)
        
    #hsv_2[:, :, 2] = np.where((hsv_2[:, :, 0]>6) & (hsv_2[:, :, 0]<30) ,hsv_2[:, :, 1] *0.7,hsv_2[:, :, 2]) #黒色
    #0.001 赤
    #0.3 緑


    #bgr = cv2.cvtColor(hsv_3, cv2.COLOR_HSV2BGR)
    cv2.imwrite(output_path, bgr)