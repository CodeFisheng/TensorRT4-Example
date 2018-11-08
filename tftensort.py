
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow.contrib.tensorrt as trt ##  WARNNING: must load when do prediction!!!!

"""
@time:     2018-10-07 19:27
@platform: vim
@author:   YunYang1994
@email:    yyang@nullmax.ai
"""
################################################################################
# Run this script
# => Sub-Graph Optimizations within TensorFlow
################################################################################

class Trtmodel(object):
    """
    ==> apply TensorRT optimizations to the frozen graph
    """

    def __init__(self, pb_file, labels_file, input_node, output_node):
        """
        Args:
            pb_file: The location of a Frozen Graph.
            labels_file: json file, {id: label}
            input_node:  The name of the graph input node.
            output_node: The names of the graph output node.
        """

        assert pb_file.endswith(".pb")
        self.pb_file = pb_file
        self.pb_name = os.path.basename(self.pb_file)[:-3]
        self.INPUT_NODE = input_node
        self.OUTPUT_NODE = output_node

        with open(labels_file, 'r') as labels_file:
            self.labels = json.load(labels_file)

        with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
            self.frozen_graph_def = tf.GraphDef()
            self.frozen_graph_def.ParseFromString(f.read())

    def optimize(self, output_dir, mode):

        """
        Args:
            output_dir: The location of a optimized graph directory
            mode: benchmark the model with TensorRT
        """

        assert mode in ["FP16", "FP32", "INT8"]
        graph_def = trt.create_inference_graph(input_graph_def=self.frozen_graph_def,
                                               outputs=[self.OUTPUT_NODE],
                                               max_batch_size=128,
                                               max_workspace_size_bytes=1 << 32,
                                               precision_mode=mode)

        output_path = os.path.join(output_dir, self.pb_name+"-{}.pb".format(mode.lower()))
        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(graph_def.SerializeToString())


    def predict(self, image_data, verbose=True):
        """
        Args:
            image_data: np.ndarray, shape: [BATCH_SIZE, HEIGHT, WIDTH, CHANNELS]
            verbose: True or False
        """
        if verbose: import time

        _R_MEAN = 123.68 # ADJUST
        _G_MEAN = 116.78 # ADJUST
        _B_MEAN = 103.94 # ADJUST
        _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
        BATCH_SIZE, HEIGHT, WIDTH = image_data.shape[:3]

        means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
        image_data = image_data - means

        tf.reset_default_graph()
        graph = tf.Graph()

        # tf.logging.info("Starting execution")
        with graph.as_default():
            images = tf.placeholder("float", [BATCH_SIZE, HEIGHT, WIDTH, 3])
            return_tensors = tf.import_graph_def(graph_def=self.frozen_graph_def,
                                                input_map={self.INPUT_NODE: images},
                                                return_elements=[self.OUTPUT_NODE])

            # Unwrap the returned output node. For now, we assume we only
            # want the tensor with index `:0`, which is the 0th element of the
            # `.outputs` list.
            output = return_tensors[0].outputs[0]

        with tf.Session(graph=graph) as sess:
            embeddings = sess.run([output], feed_dict={images:image_data})

            start = time.time()
            result = [self.labels[str(np.argmax(embedding[0]))] for embedding in embeddings]
            if verbose: print("=> running time: {} prediction: {}".format(time.time()-start, result))

        return result

if __name__ == "__main__":
    pass
    # from scipy import misc
    ### TODO
    # #----------------------------- Op type registered -------------------------#
    # ### => contrib ops are lazily registered when the module is first accessed.
    # ### => first registered process!
    # ### => Sub-Graph Optimizations within TensorFlow

    # mod = "FP16"
    # model = Trtmodel("./model/resnetv2.pb", "./data/labellist.json", "input_tensor", "softmax_tensor")
    # model.optimize("./model", mod)
    # model = Trtmodel("./model/resnetv2-fp16.pb", "./data/labellist.json", "input_tensor", "softmax_tensor")
    # image_data = misc.imread("./data/image.jpg")
    # image_data = np.tile(image_data, [1, 1, 1, 1])
    # model.predict(image_data)
    # #----------------------------- Op type registered -------------------------#
