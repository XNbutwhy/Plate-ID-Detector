from paddleocr import PaddleOCR,draw_ocr
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
import PIL
from PIL import Image
import math
import torch

class Plate_Recognition(Object):
    def __init__(self, input_size, yolo_weights, det_th, nms_th, class_name, ocr_det, ocr_recog, lang='en'):
        # __init__ yolov8 detection
        self.yolo_weights = yolo_weights

        # load detection model
        self.yolo_model = load_model(self.yolo_weights, map='cpu')
        self.ocr_model = ocr.ocr('image_path')


    def __call__(self, img, save_path):
        resize_img = self._preprocess(img) #, input_size
        # Load a pretrained YOLOv8n model
        det_model = YOLO('/content/drive/MyDrive/BienSoXe/ultralytics/runs/detect/train14/weights/best.pt')

        # Perform inference on an image
        result_0 = det_model(img)

        # Extract bounding boxes, classes, names, and confidences
        boxes_0 = result_0[0].boxes.xyxy.tolist()
        # classes = result_0[0].boxes.cls.tolist()
        # names = result_0[0].names
        score = result_0[0].boxes.conf.tolist()


        # loading the image
        img = PIL.Image.open(img)

        # fetching the dimensions
        wid, hgt = img.size

        # displaying the dimensions
        print(str(wid) + "x" + str(hgt))


        # Crop image
        box = (math.floor(boxes_0[0][0]), math.floor(boxes_0[0][1]), math.ceil(boxes_0[0][2]), math.ceil(boxes_0[0][3]))
        cropped_image = img.crop(box)
        final_path = save_path + '/cropped_image.jpg'
        cropped_image.save(final_path)

        result_1 = ocr.ocr(final_path)

        # run save_ocr
        end_path = os.path.join(save_path, img.split('/')[-1] + 'output.jpg')

        image = cv2.imread(img)
        boxes = [None] * len(result_1[0])
        txts = [None] * len(result_1[0])
        scores = [None] * len(result_1[0])
        for idx, line in enumerate(result_1[0]):

            boxes[idx] = line[0]
            txts[idx] = line[1][0]
            scores[idx] = line[1][1]


        # detect plate obj (copy paste)

        # read text (copy paste)

        return box, score, txts, cropped_image, end_path
    
    def _preprocess(self, img): #, input_size
        not_tensor = not isinstance(img, torch.Tensor)
        if not_tensor:
            img = np.stack(self.pre_transform(img))
            img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            img = np.ascontiguousarray(img)  # contiguous
            img = torch.from_numpy(img)

        img = img.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img /= 255  # 0 - 255 to 0.0 - 1.0
        return img
    
    def _postprocess(self, box, score, txts):
        # return, print plate contents (copy + paste)
        print(box, score, txts)
    
    def visualize(self, cropped_image, boxes, txts, scores, font, save_path):
        im_show = draw_ocr(cropped_image, boxes, txts, scores, font_path=font)
        print(boxes)
        print(txts)
        print(scores)
        cv2.imwrite(save_path, im_show)

        img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
