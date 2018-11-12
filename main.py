import sys
import tftensort
import argparse



def main(argv):

    flags = parser(description="Parser to contain flags for running the TensorRT timers.").parse_args()
    model = tftensort.Trtmodel(flags.frozen_graph, flags.labels_file, flags.input_node, flags.output_node)
    # BATCH_SIZE = flags.batch_size

    if flags.fp16:
        model.optimize(flags.output_dir, mode='FP16')
    if flags.fp32:
        model.optimize(flags.output_dir, mode='FP32')
    if flags.int8:
        model.optimize(flags.output_dir, mode='INT8')

    if flags.image_file is not None:
        model.predict(flags.image_file)

class parser(argparse.ArgumentParser):

    def __init__(self,description):
        super(parser, self).__init__(description)

        self.add_argument(
            "--frozen_graph", "-fg", default=None,
            help="[default: %(default)s] The location of a Frozen Graph ",
            metavar="<FG>",
        )

        self.add_argument(
            "--labels_file", "-lf", default="./data/labellist.json",
            help="[default: %(default)s] The location of a labels_file ",
            metavar="<LF>",
        )

        self.add_argument(
            "--output_dir", "-od", default="./model",
            help="[default: %(default)s] The location where output files will "
            "be saved.",
            metavar="<OD>",
        )

        self.add_argument(
            "--input_node", "-in", default="input_tensor",
            help="[default: %(default)s] The name of the graph input node where "
            "the float image array should be fed for prediction.",
            metavar="<IN>",
        )

        self.add_argument(
            "--output_node", "-on", default="softmax_tensor",
            help="[default: %(default)s] The names of the graph output node "
            "that should be used when retrieving results. Assumed to be a softmax.",
            metavar="<ON>",
        )

        self.add_argument(
            "--image_file", "-if", default=None,
            help="[default: %(default)s] The location of a JPEG image that will ",
            metavar="<IF>",
        )

        self.add_argument(
            "--fp32", action="store_true",
            help="[default: %(default)s] If set, benchmark the model with TensorRT "
            "using fp32 precision."
        )

        self.add_argument(
            "--fp16", action="store_true",
            help="[default: %(default)s] If set, benchmark the model with TensorRT "
            "using fp16 precision."
        )

        self.add_argument(
            "--int8", action="store_true",
            help="[default: %(default)s] If set, benchmark the model with TensorRT "
            "using int8 precision."
        )

if __name__ == "__main__": main(argv=sys.argv)
