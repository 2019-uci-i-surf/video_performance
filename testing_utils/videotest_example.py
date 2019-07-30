import keras
import pickle
from videotest import VideoTest

import sys
sys.path.append("..")

model_name="MobileNetSSD300"

if model_name=="MobileNetSSD300":
    from model.ssd300MobileNet import SSD
    weight_name = 'mobilenetssd300.hdf5'

input_shape = (300,300,3)

# Change this if you run with other classes than VOC
class_names = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];
NUM_CLASSES = len(class_names)

model = SSD(input_shape, num_classes=NUM_CLASSES)

# Change this path if you want to use your own trained weights
model.load_weights('./../weights/'+weight_name)
        
vid_test = VideoTest(class_names, model, input_shape)

# To test on webcam 0, remove the parameter (or change it to another number
# to test on that webcam)
vid_test.run('./../videos/Walk - 20454.mp4')
