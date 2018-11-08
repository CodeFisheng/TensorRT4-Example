import tftensort
import numpy as np
from scipy import misc

# model = tftensort.Trtmodel("./model/resnetv2.pb", "./data/labellist.json", "input_tensor", "softmax_tensor")
model = tftensort.Trtmodel("./model/resnetv2-fp16.pb", "./data/labellist.json", "input_tensor", "softmax_tensor")
# model.optimize("./model", "FP16")
image_data = misc.imread("./data/image.jpg")
image_data = np.tile(image_data, [1, 1, 1, 1])
model.predict(image_data)



