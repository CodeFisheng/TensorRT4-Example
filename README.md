# Running the TensorFlow Official ResNet with TensorRT 4

NVIDIA announced the integration of TensorRT 4.0.1.6 inference optimization tool with TensorFlow. TensorRT integration could be available for use in the module `contrib` of TensorFlow. In this script, TensorRT rewrites parts of the execution graph to allow for faster prediction times. The following figure details the workflow of tensorflow graph optimizaition in TensorRT.

<p align="center">
    <img width="70%" src="https://github.com/YunYang1994/Pytensort/blob/master/image/workflow.png" style="max-width:90%;">
    </a>
</p>
