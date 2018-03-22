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

def load_graph(path_to_ckpt):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    return detection_graph


def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]

  return output_dict

def detect(img, my_graph, threshold=0.5):

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
    output_dict = run_inference_for_single_image(image_passed, my_graph)
    # Visualization of the results of a detection.

    print time.time() - start_time

    output_dict['detection_scores'] = output_dict['detection_scores'][output_dict['detection_scores']>threshold]
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0:len(output_dict['detection_scores']), :]
    output_dict['detection_classes'] = output_dict['detection_classes'][0:len(output_dict['detection_scores'])]
    output_dict['num_detections'] = len(output_dict['detection_scores'])
    print output_dict

    return output_dict
