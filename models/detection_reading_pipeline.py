from .yolo_detector import YOLODetector
from .numberplate_text_reader import NumberPlateTextReader

# from numberplate_classifier import NumberPlateClassifier
from multiprocessing import Queue

import time
import cv2

import numpy as np
from typing import List
import math


def fline(p0: List, p1: List, debug: bool = False) -> List:
    """
    Вычесление угла наклона прямой по 2 точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    if debug:
        print("Уравнение прямой, проходящей через эти точки:")
    if x1 - x2 == 0:
        k = math.inf
        b = y2
    else:
        k = (y1 - y2) / (x1 - x2)
        b = y2 - k * x2
    if debug:
        print(" y = %.4f*x + %.4f" % (k, b))
    r = math.atan(k)
    a = math.degrees(r)
    a180 = a
    if a < 0:
        a180 = 180 + a
    return [k, b, a, a180, r]


def distance(p0: List or np.ndarray, p1: List or np.ndarray) -> float:  # type: ignore
    """
    distance between two points p0 and p1
    """
    return math.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2)


def linear_line_matrix(p0: List, p1: List, verbode: bool = False) -> np.ndarray:
    """
    Вычесление коефициентов матрицы, описывающей линию по двум точкам
    """
    x1 = float(p0[0])
    y1 = float(p0[1])

    x2 = float(p1[0])
    y2 = float(p1[1])

    matrix_a = y1 - y2
    matrix_b = x2 - x1
    matrix_c = x2 * y1 - x1 * y2
    if verbode:
        print("Уравнение прямой, проходящей через эти точки:")
        print("%.4f*x + %.4fy = %.4f" % (matrix_a, matrix_b, matrix_c))
        print(matrix_a, matrix_b, matrix_c)
    return np.array([matrix_a, matrix_b, matrix_c])


def get_y_by_matrix(matrix: np.ndarray, x: float) -> np.ndarray:
    """
    TODO: describe function
    """
    matrix_a = matrix[0]
    matrix_b = matrix[1]
    matrix_c = matrix[2]
    if matrix_b != 0:
        return (matrix_c - matrix_a * x) / matrix_b


def find_distances(points: np.ndarray or List) -> List:  # type: ignore
    """
    TODO: describe function
    """
    distanses = []
    cnt = len(points)

    for i in range(cnt):
        p0 = i
        if i < cnt - 1:
            p1 = i + 1
        else:
            p1 = 0
        distanses.append(
            {
                "d": distance(points[p0], points[p1]),
                "p0": p0,
                "p1": p1,
                "matrix": linear_line_matrix(points[p0], points[p1]),
                "coef": fline(points[p0], points[p1]),
            }
        )
    return distanses


def build_perspective(img: np.ndarray, rect: list, w: int, h: int) -> List:
    """
    image perspective transformation
    """
    img_h, img_w, img_c = img.shape
    if img_h < h:
        h = img_h
    if img_w < w:
        w = img_w
    pts1 = np.float32(rect)
    pts2 = np.float32(np.array([[0, 0], [w, 0], [w, h], [0, h]]))
    moment = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, moment, (w, h))


def get_cv_zone_rgb(
    img: np.ndarray,
    rect: list,
    gw: float = 0,
    gh: float = 0,
    coef: float = 4.6,
    auto_width_height: bool = True,
) -> List:
    """
    TODO: describe function
    """
    if gw == 0 or gh == 0:
        distanses = find_distances(rect)
        h = (distanses[0]["d"] + distanses[2]["d"]) / 2
        if auto_width_height:
            w = int(h * coef)
        else:
            w = (distanses[1]["d"] + distanses[3]["d"]) / 2
    else:
        w, h = gw, gh
    return build_perspective(img, rect, int(w), int(h))


def contour_to_rectangle(contour):
    # Вычисляем контур с четырьмя углами, используя алгоритм Дугласа-Пекера
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # Если результат - не 4 точки, используем минимальный поворотный прямоугольник
    if len(approx) != 4:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        approx = np.int0(box)

    return approx


# Load the image from the correct path


def sort_points(points):
    # Сортируем точки по оси Y
    points_sorted_by_y = sorted(points, key=lambda point: point[1])

    # Нижние две точки (bottom_left и bottom_right)
    bottom_points = points_sorted_by_y[:2]
    # Верхние две точки (top_left и top_right)
    top_points = points_sorted_by_y[2:]

    # Сортируем нижние точки по оси X, чтобы определить левую и правую
    bottom_left, bottom_right = sorted(bottom_points, key=lambda point: point[0])

    # Сортируем верхние точки по оси X, чтобы определить левую и правую
    top_left, top_right = sorted(top_points, key=lambda point: point[0])

    return [bottom_left, bottom_right, top_right, top_left]


def find_np(image):

    blurred = cv2.GaussianBlur(image, (17, 17), 0)
    gray_image = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(
        gray_image,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,  # Или cv2.ADAPTIVE_THRESH_MEAN_C
        cv2.THRESH_BINARY,
        11,  # Размер блока для расчета порога
        2,  # Константа, вычитаемая из среднего значения
    )
    contours, _ = cv2.findContours(
        adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Assuming the largest contour is the number plate, sort by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    contour = contour_to_rectangle(contour=contours[0])
    try:
        contour = contour.squeeze(1)
    except:
        pass
    points = sort_points(contour)
    transformed_image = get_cv_zone_rgb(image, points)
    img = image.copy()
    cv2.polylines(img, [contour], isClosed=True, color=(0, 255, 0), thickness=2)
    image1_resized = cv2.resize(
        img, (transformed_image.shape[1], transformed_image.shape[0])
    )
    return transformed_image, image


class DetectionReadingPipeline:
    """
    Pipeline for numberplate detector and text reader

    """

    def __init__(
        self,
        q_in: Queue,
        q_out: Queue,
        path_to_np: str,
        path_to_reader: str,
        reader_dict: str,
        q_interim: Queue,
    ):
        self.q_in = q_in
        """input queue"""
        self.q_out = q_out
        """output queue"""
        self.q_interim = q_interim
        """queue for results of numberplate  detector that text reader uses"""
        self.path_to_np = path_to_np
        """path to numberplate detector model"""
        self.path_to_reader = path_to_reader
        """path to reader weights"""
        self.reader_dict = reader_dict
        """dictionary for reader"""

    def yolo_run(self):
        """
        Run numberplate detector and send results to test reader
        """
        yolo = YOLODetector(self.path_to_np)

        while True:
            if self.q_in.empty():
                time.sleep(0.001)
                continue
            if self.q_interim.full():
                continue
            if self.q_out.full():
                continue
            frame, frame_id, car_id = self.q_in.get()

            result = yolo.run(frame)

            if len(result) != 0:
                x1, y1, x2, y2 = result[0][:4]

                number_plate_crop = frame[y1:y2, x1:x2].copy()
                if number_plate_crop.shape[0] == 0 or number_plate_crop.shape[1] == 0:
                    self.q_out.put(([], ["", 0], frame_id, car_id, []))
                    continue
            else:
                self.q_out.put(([], ["", 0], frame_id, car_id, []))
                continue
            self.q_interim.put((result, number_plate_crop, frame_id, car_id))

    def ppocr_run(self):
        """
        Run text reader and send output to q_out
        """
        ocr = NumberPlateTextReader(self.path_to_reader, self.reader_dict)
        while True:
            if self.q_interim.empty():
                time.sleep(0.001)
                continue
            if self.q_out.full():
                time.sleep(0.001)
                continue
            result, number_plate_crop, frame_id, car_id = self.q_interim.get()

            text = ocr(number_plate_crop)
            if text[1] < 0.9:
                fixed_crop, resized_crop = find_np(number_plate_crop)
                res = ocr(fixed_crop)

                self.q_out.put((result, res, frame_id, car_id, number_plate_crop))
            else:

                self.q_out.put((result, text, frame_id, car_id, number_plate_crop))
