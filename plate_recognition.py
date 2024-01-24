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
    
    def Video_Inference(self, video, save_path):
        # Open the video file
        cap = cv2.VideoCapture(video)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Define the codec and create VideoWriter object
        output_video_path = save_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Process each frame
        while True:
            ret, frame = cap.read()

            if not ret:
                break  # Break the loop if the video is over

            # Draw a rectangle on the frame
            # Rectangle parameters: (image, start_point, end_point, color, thickness)
            result, rect = self.Image_Inference(frame)
            # Write the frame to the output video
            out.write(rect)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Release the video capture and writer objects
        cap.release()
        out.release()

        # Close all OpenCV windows
    def Image_Inference(self, img, save_img=False):
    # video > result
        """
        Args: img (BGR) numpy
        Returns: plate_str (list of N str value): value of license plate
        ... --> save box and image output (if needed)
        """
        
        # im = img.copy()
        # Preprocess
        im = self._preprocess(img)
        detect_output = self.detect(im)
        # Detect plate boxes
        result, boxes_det, score = self._postprocess(detect_output, im, [img])
        box = boxes_det.numpy()
        txts = []
        imx = img
        # Read plate by OCR
        for i in range(box.shape[0]):
            tmp_box = box[i, :4]
            print(tmp_box)
            print(type(tmp_box))
            x1 = int(tmp_box[0])
            y1 = int(tmp_box[1])
            x2 = int(tmp_box[2])
            y2 = int(tmp_box[3])
            
            cropped_image = img[y1:y2, x1:x2]
            result_ocr = self.ocr_model.ocr(cropped_image)
            
            if result_ocr[0] is None:
                break
            print(result_ocr)
            boxes = [None] * len(result_ocr[0])
            txts = [None] * len(result_ocr[0])
            scores = [None] * len(result_ocr[0])

            imx = cv2.rectangle(img, (x1, y1), (x2, y2), color = (0, 255, 0), thickness = 2)

            
            
            for idx, line in enumerate(result_ocr[0]):

                boxes[idx] = line[0]
                txts[idx] = line[1][0]
                scores[idx] = line[1][1]
                
            imx = cv2.putText(imx, ' '.join(txts), (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, False)
            plt.imshow(imx)
            plt.show()
            
            # Write output and save
            if save_img:

                out_im = self.visualize(cropped_image, boxes_det, txts, scores, self.font, end_path, image)
                pass
            
        return txts, imx
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
        image = cv2.imread(img) # BGR image
        img_resize = self._preprocess(image) #, input_size
        ocr = PaddleOCR(use_angle_cls=True, lang='en')
        # Load a pretrained YOLOv8n model
        det_model = YOLO('yolo_weights.pt')
        

        # Perform inference on an image
        result_0 = self.detect(img_resize)
        # Extract bounding boxes, classes, names, and confidences
        
        box_det, boxes_det, score = self._postprocess(result_0, img_resize, [image])

        # REplace img by image line  image = cv2.imread(img) # BGR image
        # Check carefully: BGR / RGB / shape

        # Crop image
        box = boxes_det.numpy() # boxes_det has N boxes pred[:, :4] --> pred.shape[0] = N
        # for i in range N: --> tmp_box = pred[i,:4] ....
        x1 = int(box[0][0])
        y1 = int(box[0][1])
        x2 = int(box[0][2])
        y2 = int(box[0][3])
        cropped_image = image[y1:y2, x1:x2]
        # final_path = save_path + '/cropped_image.jpg'
        
        result_1 = self.ocr_model.ocr(cropped_image)

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

        self.visualize(cropped_image, boxes_det, txts, scores, self.font, end_path, image)
        # detect plate obj (copy paste)

        return box, txts, cropped_image, end_path # score

    def _preprocess(self, im): #, input_size

        """
        Args: Input : im <numpy array>: HWC
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
            img = img[None]  # expand for batch dim (CHW -> BCHW)

        return img

    def _postprocess(self, preds, img, orig_imgs):
        """
        Post-processes predictions and returns a list of Results objects.
        Args:
                preds
                img: resized image from preprocessing
                orig_imgs: input image
        
        Returns:
                results
                pred[:, :4]: bounding boxes
                score: confidence score
        """
        
        preds = ops.non_max_suppression(preds,
                                        conf_thres=0.25,
                                        iou_thres=0.45,
                                        max_det=100)

        # print(preds, len(preds))
        # if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            # orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        # print(len(orig_imgs))

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i]
            # print(pred[:, :4])
            score = pred[:, 4]
            print(score)
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            results.append(Results(orig_img, path=None, names=self.yolo_model.names, boxes=pred))
        # print(img.shape[2:], pred[:, :4], orig_img.shape)
        return results, pred[:, :4], score
    
    def visualize(self, cropped_img, boxes, txts, scores, font, save_path, original):
        boxes = boxes.numpy()
        x1 = int(boxes[0][0])
        y1 = int(boxes[0][1])
        x2 = int(boxes[0][2])
        y2 = int(boxes[0][3])

        imx = cv2.rectangle(original, (x1, y1), (x2, y2), color = (0, 255, 0))
        imx = cv2.putText(imx, txts[0], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA, False)
        plt.imshow(imx)
        plt.show()
        
        # cv2cropped = cv2.imread(cropped_img)
        im_show = draw_ocr(cropped_img, boxes, txts, scores, font_path=font)
        print(boxes)
        print(txts)
        print(scores)
        cv2.imwrite(save_path, im_show)

        img = cv2.cvtColor(im_show, cv2.COLOR_BGR2RGB)
        plt.imshow(img)