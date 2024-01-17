from PaddleOCR import PaddleOCR,draw_ocr
import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np
from yolov8.ultralytics import YOLO
import PIL
from PIL import Image
import math
import torch
from yolov8.ultralytics.nn.autobackend import AutoBackend
from yolov8.ultralytics.utils import ASSETS
# from yolov8.ultralytics.models.yolo.detect import DetectionPredictor
from yolov8.ultralytics.data.augment import LetterBox
from yolov8.ultralytics.utils.checks import check_imgsz, check_imshow
from yolov8.ultralytics.engine.results import Results
from yolov8.ultralytics.utils import ops

class Plate_Recognition():
    
    def __init__(self, yolo_weights="yolo_weights.pt", imgsz=640, device='cpu'): # det_th, nms_th, class_name, ocr_det, ocr_recog, lang='en'
        # __init__ yolov8 detection
        self.imgsz = imgsz
        if device == 'cpu':
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda')
        self.yolo_weights = yolo_weights
        self.yolo_model = AutoBackend(weights=self.yolo_weights,
                                      device=self.device,)
        self.yolo_model.eval()
        self.ocr_model = PaddleOCR(use_angle_cls=True, lang='en')
        self.font = 'PaddleOCR/simfang.ttf'
        # print(self.yolo_model)

    def detect(self, im):
        
        """
        Use YOLOv8 model to detect plate:
        args:
        im (BCHW tensor image)
        
        return:
        boxes (list): object, format: [x, y, w, h]
        """
        
        return self.yolo_model(im)

    def __call__(self, img, save_path):
        image = cv2.imread(img)
        img_resize = self._preprocess(image) #, input_size
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # Load a pretrained YOLOv8n model
        det_model = YOLO('yolo_weights.pt')
        

        # Perform inference on an image
        result_0 = self.detect(img_resize)
        # Extract bounding boxes, classes, names, and confidences
        
        box_det, boxes_det, score = self._postprocess(result_0, img_resize, [image])

        
        # boxes_0 = result_0[0].box.xyxy.tolist()
        # score = result_0[0].boxes.conf.tolist()
        

        # loading the image
        img = PIL.Image.open(img)

        # fetching the dimensions
        wid, hgt = img.size

        # displaying the dimensions
        print(str(wid) + "x" + str(hgt))


        # Crop image
        box = boxes_det.numpy()
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[0][2])
        y2 = int(box[0][3])
        cropped_image = img.crop((x1, y1, x2, y2))
        final_path = save_path + '/cropped_image.jpg'
        cropped_image.save(final_path)
        print(f"Plate image - Saved to {final_path}. Notice that the image will be replaced on the next iteration.")
        
        result_1 = ocr.ocr(final_path)

        # run save_ocr
        end_path = os.path.join(save_path + '/', 'output.jpg')
        print(end_path)

        
        boxes = [None] * len(result_1[0])
        txts = [None] * len(result_1[0])
        scores = [None] * len(result_1[0])
        for idx, line in enumerate(result_1[0]):

            boxes[idx] = line[0]
            txts[idx] = line[1][0]
            scores[idx] = line[1][1]

        self.visualize(final_path, boxes_det, txts, scores, self.font, end_path, image)
        # detect plate obj (copy paste)

        return box, txts, cropped_image, end_path # score

    def _preprocess(self, im): #, input_size

        """
        Args: 
                im <numpy array>: HWC
                imgsz: output image size (return im <imgsz imgsz>)
        Return:
                img (float 0-1): BCHW
        """

        letterbox = LetterBox(new_shape=(self.imgsz, self.imgsz))
        img = letterbox(image=im)
        img = img[..., ::-1].transpose((2, 0, 1)) # BGR -> RGB; HWC -> CHW (this is for 1 image)
        img = np.ascontiguousarray(img) # convert, ensure img = np.array format
        img = torch.from_numpy(img).to(self.device).float() # convert to torch + send to device + convert from uint8 > float
        img /= 255 # normalize
        if len(img.shape) == 3:
            print(img.shape)
            img = img[None]  # expand for batch dim (CHW -> BCHW)
            print(img.shape)

        return img

    def _postprocess(self, preds, img, orig_imgs):
        """
        Post-processes predictions and returns a list of Results objects.
        Args:
                preds: i forgor
                img: resized image from preprocessing
                orig_imgs: input image
        
        Return:
                results: i forgor
                pred[:, :4]: bounding boxes
                score: confidence score
        """
        
        preds = ops.non_max_suppression(preds,
                                        conf_thres=0.25,
                                        iou_thres=0.45,
                                        max_det=100)

        print(preds, len(preds))
        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            # orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        # print(len(orig_imgs))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            print(pred[:, :4])
            score = pred[:, 4]
            print(score)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=None, names=self.yolo_model.names, boxes=pred))
        print(img.shape[2:], pred[:, :4], orig_img.shape)
        return results, pred[:, :4], score
    
    def visualize(self, cropped_path, boxes, txts, scores, font, save_path, original):
        boxes = boxes.numpy()
        x1 = int(boxes[0][0])
        y1 = int(boxes[0][1])
        x2 = int(boxes[0][2])
        y2 = int(boxes[0][3])

        imx = cv2.rectangle(original, (x1, y1), (x2, y2), color = (0, 255, 0))
        plt.imshow(imx)
        
        cv2cropped = cv2.imread(cropped_path)
        im_show = draw_ocr(cv2cropped, boxes, txts, scores, font_path=font)
        print(boxes)
        print(txts)
        print(scores)
        cv2.imwrite(save_path, im_show)

        img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
        plt.imshow(img)