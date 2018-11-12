# Running the TensorFlow Official ResNet with TensorRT 4

NVIDIA announced the integration of TensorRT 4.0.1.6 inference optimization tool with TensorFlow. TensorRT integration could be available for use in the module `contrib` of TensorFlow. In this script, TensorRT rewrites parts of the execution graph to allow for faster prediction times. The following figure briefly details the workflow of tensorflow graph optimizaition in TensorRT.

<p align="center">
    <img width="70%" src="https://github.com/YunYang1994/Pytensort/blob/master/image/workflow.png" style="max-width:80%;">
    </a>
</p>

## How to Run This Script
### Step 1: Install Prerequisites
#### Install TensorFlow.
```bashrc
yang@yuhy:~$ pip install tensorflow-gpu==1.11.0
```
#### Install TensorRT.
download [TensorRT4](https://developer.nvidia.com/nvidia-tensorrt-download) and install it as [Installation Guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html)

### Step 2: Get a model to optimize

You can download the ResNetv2-ImageNet [Frozen Graph](http://download.tensorflow.org/models/official/resnetv2_imagenet_frozen_graph.pb) and put it in the dir `./model`, then <br>

```bashrc
yang@yuhy:~$ python3 main.py -fg ./model/resnetv2_imagenet_frozen_graph.pb --fp16
```
For the full set of possible parameters, you can run `python main.py --help`


### Step 3: Get an image to test
The script can accept a JPEG image file to use for predictions. If none is provided, We provide a sample `./data/elephent.jpg `here which can be passed in with the --image_file flag.


### Step 4: Compare inference performance with native model
```bashrc
yang@yuhy:~$ python3 main.py -fg ./model/resnetv2_imagenet_frozen_graph.pb -if ./data/elephent.jpg
```
```bashrc
yang@yuhy:~$ python3 main.py -fg ./model/resnetv2_imagenet_frozen_graph-fp16.pb -if ./data/elephent.jpg
```



