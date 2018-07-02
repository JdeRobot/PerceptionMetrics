import numpy as np
import sys

if not hasattr(sys, 'argv'):
    sys.argv  = ['']



import tensorflow as tf
import time

np.set_printoptions(threshold='nan')

if tf.__version__ < '1.4.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# What model to download.

class TensorFlowDetector:

    def __init__(self, path_to_ckpt):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
          od_graph_def = tf.GraphDef()
          with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes =  detection_graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=detection_graph)

	print "Initializing"

	dummy_tensor = np.zeros((1,1,1,3), dtype=np.int32)
        self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                               feed_dict={self.image_tensor: dummy_tensor})


    def run_inference_for_single_image(self, image):
          # sess = tf.Session(graph=d_graph)
          # Get handles to input and output tensors
          # ops = d_graph.get_operations()
          # all_tensor_names = {output.name for op in ops for output in op.outputs}
          # tensor_dict = {}
          #for key in [
            #  'num_detections', 'detection_boxes', 'detection_scores',
            #  'detection_classes'
          #]:
            #tensor_name = key + ':0'
            #if tensor_name in all_tensor_names:
             # tensor_dict[key] = d_graph.get_tensor_by_name(
            #      tensor_name)
          #image_tensor = d_graph.get_tensor_by_name('image_tensor:0')

          # Run inference

          image.setflags(write=1)
          image_expanded = np.expand_dims(image, axis=0)

          (boxes, scores, classes, num) = self.sess.run(
                                [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
                                feed_dict={self.image_tensor: image_expanded})

          return boxes[0], scores[0], classes[0].astype(np.uint8), num[0]

    def detect(self, img, threshold=0.5):

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

        boxes, scores, classes, num = self.run_inference_for_single_image(image_passed)
        # Visualization of the results of a detection.



        print "Inference Time: " + str(time.time() - start_time) + " seconds"

        output_dict = {}

        output_dict['detection_scores'] = scores[scores>threshold]
        output_dict['detection_boxes'] = boxes[0:len(output_dict['detection_scores']), :]
        output_dict['detection_classes'] = classes[0:len(output_dict['detection_scores'])]
        output_dict['num_detections'] = len(output_dict['detection_scores'])

        return output_dict
