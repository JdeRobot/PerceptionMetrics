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


    def reframe_box_masks_to_image_masks(box_masks, boxes, image_height,
                                     image_width):
        """Transforms the box masks back to full image masks.

        Embeds masks in bounding boxes of larger masks whose shapes correspond to
        image shape.

        Args:
        box_masks: A tf.float32 tensor of size [num_masks, mask_height, mask_width].
        boxes: A tf.float32 tensor of size [num_masks, 4] containing the box
               corners. Row i contains [ymin, xmin, ymax, xmax] of the box
               corresponding to mask i. Note that the box corners are in
               normalized coordinates.
        image_height: Image height. The output mask will have the same height as
                      the image height.
        image_width: Image width. The output mask will have the same width as the
                     image width.

        Returns:
        A tf.float32 tensor of size [num_masks, image_height, image_width].
        """

        def reframe_box_masks_to_image_masks_default():
            """The default function when there are more than 0 box masks."""
            def transform_boxes_relative_to_boxes(boxes, reference_boxes):
                boxes = tf.reshape(boxes, [-1, 2, 2])
                min_corner = tf.expand_dims(reference_boxes[:, 0:2], 1)
                max_corner = tf.expand_dims(reference_boxes[:, 2:4], 1)
                transformed_boxes = (boxes - min_corner) / (max_corner - min_corner)
                return tf.reshape(transformed_boxes, [-1, 4])

            box_masks_expanded = tf.expand_dims(box_masks, axis=3)
            num_boxes = tf.shape(box_masks_expanded)[0]
            unit_boxes = tf.concat(
                [tf.zeros([num_boxes, 2]), tf.ones([num_boxes, 2])], axis=1)
            reverse_boxes = transform_boxes_relative_to_boxes(unit_boxes, boxes)
            return tf.image.crop_and_resize(
                image=box_masks_expanded,
                boxes=reverse_boxes,
                box_ind=tf.range(num_boxes),
                crop_size=[image_height, image_width],
                extrapolation_value=0.0)
        image_masks = tf.cond(
          tf.shape(box_masks)[0] > 0,
          reframe_box_masks_to_image_masks_default,
          lambda: tf.zeros([0, image_height, image_width, 1], dtype=tf.float32))

        return tf.squeeze(image_masks, axis=3)




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

        output_dict = self.run_inference_for_single_image(image_passed)
        # Visualization of the results of a detection.



        print "Inference Time: " + str(time.time() - start_time) + " seconds"

        new_dict = {}

        new_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>threshold]
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
