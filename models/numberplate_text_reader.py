import numpy as np
import cv2
import os
from multiprocessing import Queue
import onnxruntime as ort
import math
class NumberPlateTextReader():
    """
    Text reader for numberplates
    """
    def __init__(self, path_to_model="ppocr_v3_slim_int8.rknn", path_to_dict="../../dicts/russian_v2.txt"):
        self.session = ort.InferenceSession(
            path_to_model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        """An onnx session to launch the model."""

        self.input_name: str = self.session.get_inputs()[0].name
        """The name of the input metadata."""
        self.output_name: list[str] = [self.session.get_outputs()[0].name]
        """The name of the output metadata."""

        with open(path_to_dict, "r") as f:
            self.dict = f.readlines()
        self.dict = [char.rstrip() for char in self.dict]
        self.dict = [""] + self.dict
        """dictionary for text recognition model"""
        self.input_tensor = self.session.get_inputs()[0]
        
    def preprocess(self, img):
        """Adjusts the dimensions of the original image to those required
        for processing without changing the aspect ratio.

        Parameters
        ----------
        input_img : ndarray
            The original image.

        Returns
        -------
        padding_im : ndarray
            The resized image.
        
        """
        imgC, imgH, imgW = [3, 32, 128]
        max_wh_ratio = imgW / imgH
        imgW = int((imgH * max_wh_ratio))
        w = self.input_tensor.shape[3:][0]
        if isinstance(w, str):
                pass
        elif w is not None and w > 0:
            imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        padding_im = padding_im[np.newaxis, :]
        return padding_im
    
    def inference(self, img):
        """Processes the image with a neural network.

        Parameters
        ----------
        img : ndarray
            The image to be processed.

        Returns
        -------
        result_onnx : ndarray
            The results of image processing by a neural network.
        conf_list : list
            List of confidence scores for each character
        """
        outputs = self.session.run(self.output_name, {self.input_name: img})
        result_onnx = np.array([int(np.argmax(item)) for item in outputs[0][0]])
        conf_list = np.array([max(item) for item in outputs[0][0]])
        return result_onnx, conf_list
    
    def postprocess(self, result):
        """Process output of text recognition model to text
        Parameters
        ----------
        result : ndarray
            Contains outputs of model and list of confidences for each characters

        Returns
        -------
        text : str
            Predicted text
        conf : float
            Confidence score for predicted text
        """
        preds = result[0]
        conf_list = result[1]
        selection = np.ones(len(preds), dtype=bool)
        selection[1:] = preds[1:] != preds[:-1]
        char_list = [
                self.dict[text_id] for text_id in preds[selection]
            ]
        text = "".join(char_list)
        conf = np.mean(conf_list)
        return text, conf
    
    def __call__(self, img):
        """Recognize text of input image
        Parameters
        ----------
        img : ndarray
            The image to be processed.

        Returns
        -------
        prediction: str
            recognition result
        """
        img = self.preprocess(img)
        outputs = self.inference(img)
        prediction = self.postprocess(outputs)
        return prediction

if __name__ == "__main__":
    pass

    
