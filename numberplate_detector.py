import os
import numpy as np
from collections import Counter
from multiprocessing import Process, Queue, active_children
import time
import cv2
from detector import Detector
from models.detection_reading_pipeline import DetectionReadingPipeline
from models.yolo_detector import YOLODetector
from utils.util import get_crop_from_frame
from config import numberplate_cfg
import argparse


class CarData():
    """Class for storing car data and sending best results"""
    def __init__(self, resend_time : int):
        self.general = {}
        """dict for storing data"""
        self.resend_time = resend_time
        """"""
        self.general["working_time_sec"] = time.monotonic()
        self.full_reset_time = 10000
        
    def reset(self, car_id: int, task_id: str):
        """
        Resets info about given car id
        """
        self.general[task_id][car_id] = {}
        self.general[task_id][car_id]["pred_text"] = "undefined"
        self.general[task_id][car_id]["pred_list"] = []
        self.general[task_id][car_id]["pred_class"] = ""
        self.general[task_id][car_id]["class_list"] = []
        self.general[task_id][car_id]["time_passed"] = time.monotonic()
        self.general[task_id][car_id]["np_crop"] = []
        self.general[task_id][car_id]["crop_score"] = 0
        
    def update(self, data: dict, task_id: str, car_id: int):
        """update information about given car id and send best shot"""
        if task_id not in self.general:
            self.general[task_id] = {}

        if car_id not in self.general[task_id]:
            self.reset(car_id, task_id)
        self.general[task_id][car_id]["pred_list"].append(data["pred_text"])
        self.general[task_id][car_id]["class_list"].append(data["pred_class"])
        count_dict = Counter(self.general[task_id][car_id]["class_list"])
        self.general[task_id][car_id]["pred_class"] = count_dict.most_common(1)[0][0]
        count_dict = Counter(self.general[task_id][car_id]["pred_list"])
        non_unreadable_values = [value for value, count in count_dict.items() if value != "unreadable"]
        if non_unreadable_values:
            self.general[task_id][car_id]["pred_text"] = max(non_unreadable_values, key=count_dict.get) # type: ignore
        else:
            self.general[task_id][car_id]["pred_text"] = "unreadable"
        if data["pred_score"] > self.general[task_id][car_id]["crop_score"]:
            self.general[task_id][car_id]["np_crop"] = data["np_crop"]
            self.general[task_id][car_id]["crop_score"] = data["pred_score"]
        
        x1, y1, x2, y2 = data["np_bbox"]
        pred_cls = self.general[task_id][car_id]["pred_class"]
        conf = self.general[task_id][car_id]["crop_score"]
        predicted_image_text = self.general[task_id][car_id]["pred_text"]
        numberplate_crop = np.array(self.general[task_id][car_id]["np_crop"]).tolist()
        if len(self.general[task_id][car_id]["pred_list"]) == 10:
            return [x1, y1, x2, y2, int(car_id), int(pred_cls), float(conf), str(predicted_image_text), numberplate_crop]
        elif time.monotonic() - self.general[task_id][car_id]["time_passed"] > self.resend_time:
            self.general[task_id][car_id]["time_passed"] = time.monotonic()
            return [x1, y1, x2, y2, int(car_id), int(pred_cls), float(conf), str(predicted_image_text), numberplate_crop]
        if time.monotonic() - self.general["working_time_sec"] > self.full_reset_time:
            self.general["working_time_sec"] = time.monotonic()
            self.general[task_id] = {}
        return None

class NumberPlateDetector():
    """
    NumberPlate Detector class.

    Attributes
    ----------
    data_q: Queue
        queue for car crops to recognize
    inter_q: Queue
        queue for numberplate detecor and text reader interaction
    res_q: Queue
        queue for results
    yolo: YOLODetector
        car detector
    text_procs: list of Process
        processes for text recognition
    np_procs: list of Process
        processes for numberplate detection
    """
    
    def __init__(self) -> None:
        yolo_path = "./models/yolov8s.onnx"
        reader_dict = "./dicts/cis_dict.txt"
        reader_weights = "./models/ppocrv3.onnx"
        self.detect_model_path = './models/yolo_classifier.onnx'


        self.data_q = Queue(maxsize=20)
        self.res_q = Queue(maxsize=20)
        self.inter_q = Queue(maxsize=20)
        self.yolo = YOLODetector(yolo_path, classes=numberplate_cfg["car_classes"], net_size=960)
        self.car_data = CarData(resend_time=numberplate_cfg["resend_time_sec"])

        text_readers = [DetectionReadingPipeline(self.data_q, self.res_q, self.detect_model_path,reader_weights, reader_dict, self.inter_q) for _ in range(2)]
        self.text_procs = [Process(target=text_readers[i].ppocr_run) for i  in range(1)]
        self.np_procs = [Process(target=text_readers[i].yolo_run) for i  in range(len(text_readers))]
        self._start()

    def __del__(self):
        for process in active_children():
            process.terminate()

    def _start(self):
        for process in self.text_procs:
            process.start()

        for process in self.np_procs:
            process.start()

            
    @staticmethod
    def draw_results(img: np.ndarray, dets: list) -> np.ndarray:
        """
        Draws all found objects, their classes, confidence, and
        track ID on the source frame.

        Parameters
        ----------
        img : ndarray
            The original image.
        dets : list of list
            A list of all the properties of each object in the frame
            (coordinates of the bounding box, class, accuracy).

        Returns
        -------
        inf_img : ndarray
            An image with all the selected objects and their properties.
        """
        inf_img = img.copy()

        for det in dets:
            cv2.rectangle(
                img=inf_img,
                pt1=(det[0] + 10, det[1] + 10),
                pt2=(det[2] - 0, det[3] - 10),
                color=(235, 215, 50),
                thickness=3,
            )
            cv2.putText(
                img=inf_img,
                text=f"{det[7]}-({det[4]}){det[5]} - {round(det[6], 2)}",
                org=(int(det[0]) + 13, int(det[1]) - 13),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        return inf_img
    
    def run(self, frame, task_id) -> list:
        """
        Processes the image with a neural network.

        Parameters
        ----------
        frame : ndarray
            The image to be processed.
        task_id : int
            timestamp of given frame

        Returns
        -------
        dets : ndarray
            [x1, y1, x2, y2, car_id, pred_cls, conf, predicted_image_text, numberplate_crop] for each car where:
            x1, y1, x2, y2 - coordinates of numberplate bbox
            car_id - track id for processed car
            pred_cls - predicted country
            conf - confidence score
            predicted_image_text - predicted text
            numberplate_crop - crop image of a numberplate
        """
        
        detections = self.yolo.track(frame)
       
        car_crops = []
        scores = []
        cls_ids = []
        car_ids = []
        if detections is None:
            return []
        for detection in detections:
            x1, y1, x2, y2, car_id, score, class_id = detection
            car_ids.append(car_id)
            car_crops.append([x1, y1, x2, y2])
            scores.append(score)
            cls_ids.append(class_id)
            
        coor_boxes = {}
        class_ids = {}
        detection_num = len(car_crops)
        for crop_car, car_id, score, class_id in zip(car_crops, car_ids, scores, cls_ids):
            coor_boxes[car_id] = crop_car.copy()
            class_ids[car_id] = class_id
            crop_car = get_crop_from_frame(frame, crop_car)
            if np.array(crop_car).shape[0] != 0 and np.array(crop_car).shape[1]:
                self.data_q.put((crop_car, task_id, car_id))
                coor_box = coor_boxes[car_id]
                coor_box = [float(item) for item in coor_box]
            else:
                detection_num -= 1

        processed_cars = 0
        dets = []

        while processed_cars < detection_num:
            if self.res_q.empty():
                continue
            result, [predicted_image_text, conf], task_id, car_id, numberplate_crop = self.res_q.get()
            processed_cars += 1

            if len(result) != 0:
                numberplate_bbox = [int(x) for x in result[0][:4]]
                x1, y1, x2, y2 = numberplate_bbox
                x1, y1, x2, y2 = round(coor_boxes[car_id][0] + x1), round(coor_boxes[car_id][1] + y1), round(coor_boxes[car_id][0] + x2), round(coor_boxes[car_id][1] + y2)
                pred_cls = class_ids[car_id]
                data = {
                    "np_bbox" : [x1, y1, x2, y2 ],
                    "pred_class" : int(pred_cls),
                    "pred_text" : str(predicted_image_text),
                    "np_crop" : list(numberplate_crop),
                    "pred_score" : float(conf),
                }
                det = self.car_data.update(data, task_id, car_id)
                if det is not None:
                    dets.append(det)
        return dets
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        video_name = video_path.split("/")[-1].split(".")[0]
        _video_file = f'videos/{video_name}_result.mp4' # delete later

        out = cv2.VideoWriter(_video_file, fourcc, fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            dets = self.run(frame, 1)
            inf_img = self.draw_results(frame, dets)

            out.write(inf_img)
        cap.release()
        out.release()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Numberplate Detector Inference Script')
    # Basic parameters
    parser.add_argument('video_path', type=str, help='path to the video file')
    args = parser.parse_args()
    detector = NumberPlateDetector()
    detector.process_video(args.video_path)
    del detector