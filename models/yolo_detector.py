import sys
from pathlib import Path

import numpy as np
from yolox.tracker.byte_tracker import BTArgs, BYTETracker

path_root = Path(__file__).parents[2]
sys.path.append(str(path_root))

from detector import Detector
from config import numberplate_cfg
NP_CLASSES = 11


class YOLODetector(Detector):
    """
    Class for YOLO based models.
    """

    def __init__(
        self,
        path_to_model="../weights/np_detector.rknn",
        classes = [i for i in range(len(numberplate_cfg["numberplate_classes_dict"]))],
        net_size=640,
    ):

        self.net_size = net_size

        self.tracker = BYTETracker(args=BTArgs())

        super().__init__(model_path=path_to_model)
        self.CLASSES = classes


    def pre_process(
        self, input_img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        img = input_img.copy()
        img, ratio, dwdh = self.letterbox(
            img, new_shape=(self.net_size, self.net_size), auto=False
        )
        img = np.expand_dims(img, axis=0).astype("float32") / 255.0
        img = np.transpose(img, [0, 3, 1, 2])

        return img, ratio, dwdh

    def run(self, img: np.ndarray) -> list:
        """
        Processes the image with a neural network.

        Parameters
        ----------
        img : ndarray
            The image to be processed.

        Returns
        -------
        dets : ndarray
            The results of image processing by a neural network.
        """
        dets = []

        pre_img, ratio, dwdh = self.pre_process(img)

        outputs = self.inference(np.array(pre_img))

        if outputs is not None:

            boxes, classes, scores = self.post_process(outputs)

            if boxes is not None:
                boxes -= np.array(dwdh * 2)
                boxes /= ratio
                boxes = boxes.round().astype(np.int32)
                for i in range(len(classes)):
                    if classes[i] in self.CLASSES:
                        x0, y0, x1, y1 = boxes[i]
                        dets.append(
                            [x0, y0, x1, y1, float(scores[i]), int(classes[i])]
                        )

        return dets

    def track(self, img):
        """
        Processes the image with a neural network and track objects

        Parameters
        ----------
        img : ndarray
            The image to be processed.

        Returns
        -------
        dets : ndarray
            The results of tracker.
        """
        dets = self.run(img)

        if len(dets) == 0:
            return None
        tracks = self.tracker.update(
            np.array(dets), img.shape[:2], img.shape[:2]
        )
        dets = []
        for t in tracks:
            x0, y0, x1, y1 = t.tlbr
            tid = t.track_id
            cls_id = t.cls_id
            dets.append([x0, y0, x1, y1, tid, t.score, int(cls_id)])
        return dets


if __name__ == "__main__":
    pass
