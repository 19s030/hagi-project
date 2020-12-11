from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage, FlexSendMessage,CarouselContainer,BubbleContainer
import main
import json
import os
import imutils
import cv2
import matplotlib.pyplot as plt
import numpy as np
import path_data


def hair_image(event,userid):
    image_path, output_path = path_data.get_image_path(event,userid)

    image = cv2.imread(image_path)   
 