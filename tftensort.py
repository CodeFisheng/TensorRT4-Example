
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
# Requirement: tensorrt version ==> '4.0.1.6'
################################################################################

_RESIZE_MIN = 256
_R_MEAN = 123.68 # ADJUST
_G_MEAN = 116.78 # ADJUST
_B_MEAN = 103.94 # ADJUST
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
means = np.expand_dims(np.expand_dims(_CHANNEL_MEANS, 0), 0)
tf.logging.set_verbosity(tf.logging.INFO)

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

        # with tf.gfile.FastGFile(self.pb_file, 'rb') as f:
            # self.frozen_graph_def = tf.GraphDef()
            # self.frozen_graph_def.ParseFromString(f.read())
        self.frozen_graph_def = self.get_frozen_graph(self.pb_file)

    def get_frozen_graph(self, pb_file):
        """Read Frozen Graph file from disk."""
        with tf.gfile.FastGFile(pb_file, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        return graph_def

    def optimize(self, output_dir, mode):

        """
        Args:
            output_dir: The location of a optimized graph directory
            mode: benchmark the model with TensorRT
        """

        assert mode in ["FP16", "FP32", "INT8"]
        graph_def = trt.create_inference_graph(input_graph_def=self.frozen_graph_def,
                                               outputs=[self.OUTPUT_NODE],
                                               max_batch_size=1,
                                               max_workspace_size_bytes=1 << 32,
                                               precision_mode=mode)

        output_path = os.path.join(output_dir, self.pb_name+"-{}.pb".format(mode.lower()))

        with tf.gfile.GFile(output_path, 'wb') as f:
            f.write(graph_def.SerializeToString())
        return graph_def

    def _smallest_size_at_least(self, height, width, resize_min):
        resize_min = tf.cast(resize_min, tf.float32)
        height, width = tf.cast(height, tf.float32), tf.cast(width, tf.float32)

        smaller_dim = tf.minimum(height, width)
        scale_ratio = resize_min / smaller_dim

        new_height = tf.cast(height * scale_ratio, tf.int32)
        new_width = tf.cast(width * scale_ratio, tf.int32)

        return new_height, new_width

    def _resize_image(self, image, height, width):
        return tf.image.resize_images(
            image, [height, width], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)

    def _aspect_preserving_resize(self, image, resize_min):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        new_height, new_width = self._smallest_size_at_least(height, width, resize_min)

        return self._resize_image(image, new_height, new_width)

    def _central_crop(self, image, crop_height, crop_width):
        shape = tf.shape(image)
        height, width = shape[0], shape[1]

        amount_to_be_cropped_h = (height - crop_height)
        crop_top = amount_to_be_cropped_h // 2
        amount_to_be_cropped_w = (width - crop_width)
        crop_left = amount_to_be_cropped_w // 2
        return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])

    def preprocess_image(self, image_file, num_channels, output_height, output_width):

        image_buffer = tf.read_file(image_file)
        image = tf.image.decode_jpeg(image_buffer, channels=num_channels)
        image = self._aspect_preserving_resize(image, _RESIZE_MIN)
        print(image, output_height, output_width)
        image = self._central_crop(image, output_height, output_width)

        image.set_shape([output_height, output_width, num_channels])
        image = image - means

        with tf.Session() as sess:
            return sess.run([image])[0]

    def predict(self, image_file, verbose=True, num_loops=100):

        import time

        image_data = self.preprocess_image(image_file, 3, 224, 224)
        image_data = np.tile(image_data, [1, 1, 1, 1])

        tf.reset_default_graph()
        graph = tf.Graph()

        # tf.logging.info("Starting execution")
        with graph.as_default():
            images = tf.placeholder("float", [1, 224, 224, 3])
            return_tensors = tf.import_graph_def(graph_def=self.frozen_graph_def,
                                                input_map={self.INPUT_NODE: images},
                                                return_elements=[self.OUTPUT_NODE])

            # Unwrap the returned output node. For now, we assume we only
            # want the tensor with index `:0`, which is the 0th element of the
            # `.outputs` list.
            output = return_tensors[0].outputs[0]

        with tf.Session(graph=graph) as sess:
            tf.logging.info("Starting Warmup cycle")
            for _ in range(10): sess.run([output], feed_dict={images: image_data})

            tf.logging.info("Starting timing.")
            timing = []
            for _ in range(num_loops):
                start = time.time()
                embeddings = sess.run([output], feed_dict={images:image_data})
                result = [self.labels[str(np.argmax(embedding[0]))] for embedding in embeddings]
                timing.append(time.time()-start)

            tf.logging.info("Timing loop done!")
            speed = 1/np.array(timing) # BATCH_SIZE = 1
            t_max, t_min, t_mean, t_std = max(speed), min(speed), speed.mean(), speed.std()
            print("=> prediction: {}".format(result))
            print("=> Frame Per Second info: max {:.2f} fps, min {:.2f} fps, mean {:.2f} fps, std {:.2f} fps"\
                  .format(t_max, t_min, t_mean, t_std))

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
