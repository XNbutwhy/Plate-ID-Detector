from paddleocr import PaddleOCR,draw_ocr
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import PIL
from PIL import Image

class Plate_Recognition(Object):
    def __init__(self, input_size, yolo_weights, det_th, nms_th, class_name, ocr_det, ocr_recog, lang='en'):
        # __init__ yolov8 detection
        self.yolo_weights = yolo_weights

        # load detection model
        self.yolo_model = load_model(self.yolo_weights, map='cpu')
        self.ocr_model = ocr.ocr('image_path')

    def __call__(self, img):
        resize_img = self._preprocess(img, input_size)
        # detect plate obj (copy paste)

        # read text (copy paste)

        return box, score, text, cropped_image
    
    def _preprocess(self, img, input_size):
        pass
    
    def _postprocess(self, box, score, text):
        # return, print plate contents (copy + paste)
    
    def visualize(self, cropped_image):
        # draw image (copy + paste)