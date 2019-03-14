import numpy as np
import sys
from distutils.version import StrictVersion

if not hasattr(sys, 'argv'):
    sys.argv  = ['']


import tensorflow as tf
import time

np.set_printoptions(threshold=sys.maxsize)

if StrictVersion(tf.__version__) < StrictVersion('1.4.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.4.* or later!')


class TensorFlowDetector:

    def __init__(self, path_to_ckpt):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            self.tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            #if 'detection_masks' in self.tensor_dict:
        # The following processing is only for single image
                #detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                #detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                #real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                #detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                #detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                #detection_masks_reframed = self.reframe_box_masks_to_image_masks(
                #detection_masks, detection_boxes, image.shape[0], image.shape[1])
                #detection_masks_reframed = tf.cast(
                #tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
                #self.tensor_dict['detection_masks'] = detection_masks

            self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')



        #self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        #self.detection_boxes =  detection_graph.get_tensor_by_name('detection_boxes:0')
        #self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        #self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        #self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=detection_graph)

	print "Initializing"

	dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
    	self.sess.run(self.tensor_dict,
                        feed_dict={self.image_tensor: dummy_tensor})






    def run_inference_for_single_image(self, image):

        image.setflags(write=1)
        image_expanded = np.expand_dims(image, axis=0)

        output_dict = self.sess.run(
                            self.tensor_dict,
                            feed_dict={self.image_tensor: image_expanded})

        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]
        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]


        return output_dict

    def detect(self, img, threshold):

        print "Starting inference"
        start_time = time.time()

        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # image_np = load_image_into_numpy_array(image)
        image_passed = img
        #print image_np.shape
        #print image_passed.shape


        #detection_graph = load_graph(model_path)



        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)
        # Actual detection.


        start_time = time.time()


        output_dict = self.run_inference_for_single_image(image_passed)
        # Visualization of the results of a detection.



        print "Inference Time: " + str(time.time() - start_time) + " seconds"

        new_dict = {}

        new_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>=threshold]
        new_dict['detection_boxes'] = output_dict['detection_boxes'][0:len(new_dict['detection_scores']), :]
        new_dict['detection_classes'] = output_dict['detection_classes'][0:len(new_dict['detection_scores'])]
        new_dict['num_detections'] = len(new_dict['detection_scores'])
        if 'detection_masks' in output_dict:
            new_dict['detection_masks'] = output_dict['detection_masks'][0:len(new_dict['detection_scores']), :]


        #mask = new_dict['detection_masks'][0]
        #mask = mask > 0.5
        #cv2.imshow("my mask", mask)
        #cv2.waitKey(0)

        return new_dict
