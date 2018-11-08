import tensorflow as tf
from scipy import misc
import numpy as np
import tensorflow.contrib.tensorrt as trt ##  WARNNING: must load when do prediction!!!!


image_data = misc.imread("./data/image.jpg")
image_data = np.tile(image_data, [1, 1, 1, 1])

pb_file = "./model/resnetv2-fp16.pb"
with tf.gfile.FastGFile(pb_file, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())


tf.reset_default_graph()
graph = tf.Graph()

with graph.as_default():
    images = tf.placeholder("float", [1, 338, 450, 3])
    return_tensors = tf.import_graph_def(graph_def=graph_def,
                                         input_map={"input_tensor": images},
                                         return_elements=["softmax_tensor"])
    output = return_tensors[0].outputs[0]

with tf.Session(graph=graph) as sess:
    result = sess.run([output], feed_dict={images:image_data})
    print(result)






