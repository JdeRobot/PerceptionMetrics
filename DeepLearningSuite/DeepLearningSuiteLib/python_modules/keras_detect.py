import numpy as np
import sys
import time
import cv2

from PIL import Image

from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image

from keras_utils.keras_ssd_loss import SSDLoss
from keras_utils.keras_layer_AnchorBoxes import AnchorBoxes
from keras_utils.keras_layer_DecodeDetections import DecodeDetections
from keras_utils.keras_layer_L2Normalization import L2Normalization


class KerasDetector:

    def __init__(self, path_to_hdf5):

        ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

        K.clear_session() # Clear previous models from memory.

        self.model = load_model(path_to_hdf5, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                       'L2Normalization': L2Normalization,
                                                       'DecodeDetections': DecodeDetections,
                                                       'compute_loss': ssd_loss.compute_loss})

        input_size = self.model.input.shape.as_list()
        self.img_height = input_size[1]
        self.img_width = input_size[2]
        print self.img_width, self.img_height


    def detect(self, img, threshold):

        print "Starting inference"
        input_images = []

        as_image = Image.fromarray(img)
        resized = as_image.resize((self.img_width,self.img_height), Image.NEAREST)

        img_r = image.img_to_array(resized)
        input_images.append(img_r)
        input_images = np.array(input_images)

        start_time = time.time()

        y_pred = self.model.predict(input_images)

        y_pred_thresh = [y_pred[k][y_pred[k,:,1] >= threshold] for k in range(y_pred.shape[0])]

        y_thresh_array = np.array(y_pred_thresh[0])


        y_thresh_array[:, 2] /= self.img_width
        y_thresh_array[:, 3] /= self.img_height
        y_thresh_array[:, 4] /= self.img_width
        y_thresh_array[:, 5] /= self.img_height

        print "Inference Time: " + str(time.time() - start_time) + " seconds"

        return y_thresh_array
