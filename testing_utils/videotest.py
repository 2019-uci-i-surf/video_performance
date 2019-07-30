""" A class for testing a SSD model on a video file or webcam """

import cv2
import keras
from keras.applications.imagenet_utils import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
from keras.preprocessing import image
import pickle
import numpy as np
from random import shuffle
#from scipy.misc import imread, imresize
from timeit import default_timer as timer
from threading import Timer
import matplotlib.pyplot as plt
import time

import sys
sys.path.append("..")
from ssd_utils import BBoxUtility
video_path=0

class VideoTest(object):
    """ Class for testing a trained SSD model on a video file and show the
        result in a window. Class is designed so that one VideoTest object
        can be created for a model, and the same object can then be used on
        multiple videos and webcams.

        Arguments:
            class_names: A list of strings, each containing the name of a class.
                         The first name should be that of the background class
                         which is not used.

            model:       An SSD model. It should already be trained for
                         images similar to the video to test on.

            input_shape: The shape that the model expects for its input,
                         as a tuple, for example (300, 300, 3)

            bbox_util:   An instance of the BBoxUtility class in ssd_utils.py
                         The BBoxUtility needs to be instantiated with
                         the same number of classes as the length of
                         class_names.

    """

    def __init__(self, class_names, model, input_shape):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.model = model
        self.input_shape = input_shape
        self.bbox_util = BBoxUtility(self.num_classes)
        self.timer = Timer(1, self.timer_callback)
        self.current_time = 0
        self.current_fps = 0
        self.exec_time = None
        self.prev_extra_time = None
        self.extra_time = None
        self.fps_time_slot = list()
        self.is_finish = False

        # Create unique and somewhat visually distinguishable bright
        # colors for the different classes.
        self.class_colors = []
        for i in range(0, self.num_classes):
            # This can probably be written in a more elegant manner
            hue = 255*i/self.num_classes
            col = np.zeros((1,1,3)).astype("uint8")
            col[0][0][0] = hue
            col[0][0][1] = 128 # Saturation
            col[0][0][2] = 255 # Value
            cvcol = cv2.cvtColor(col, cv2.COLOR_HSV2BGR)
            col = (int(cvcol[0][0][0]), int(cvcol[0][0][1]), int(cvcol[0][0][2]))
            self.class_colors.append(col)

    def run(self, video_path, start_frame = 0, conf_thresh = 0.6):
        """ Runs the test on a video (or webcam)

        # Arguments
        video_path: A file path to a video to be tested on. Can also be a number,
                    in which case the webcam with the same number (i.e. 0) is
                    used instead

        start_frame: The number of the first frame of the video to be processed
                     by the network.

        conf_thresh: Threshold of confidence. Any boxes with lower confidence
                     are not visualized.

        """

        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError(("Couldn't open video file or webcam. If you're "
            "trying to open a webcam, make sure you video_path is an integer!"))

        # Compute aspect ratio of video
        vid.set(cv2.CAP_PROP_FPS, 30)
        vidw = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        vidh = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

        vidar = vidw/vidh

        # Skip frames until reaching start_frame
        if start_frame > 0:
            vid.set(cv2.CAP_PROP_POS_MSEC, start_frame)

        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()

        self.timer.start()
        while True:
            retval, orig_image = vid.read()

            self.exec_time = time.time()

            if not retval:
                print("Done!")
                self.timer.cancel()
                self.timer.join()
                self.exec_time = None
                avg_fps, total_time = self.calc_avg_fps(self.fps_time_slot)
                print("\naverage fps : {}, total time: {}".format(avg_fps, total_time))
                self.draw_fps(self.fps_time_slot)
                return

            im_size = (self.input_shape[0], self.input_shape[1])
            resized = cv2.resize(orig_image, im_size)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # Reshape to original aspect ratio for later visualization
            # The resized version is used, to visualize what kind of resolution
            # the network has to work with.
            to_draw = cv2.resize(resized, (int(self.input_shape[0]*vidar), self.input_shape[1]))

            # Use model to predict
            inputs = [image.img_to_array(rgb)]
            tmp_inp = np.array(inputs)
            x = preprocess_input(tmp_inp)

            y = self.model.predict(x)


            # This line creates a new TensorFlow device every time. Is there a
            # way to avoid that?
            results = self.bbox_util.detection_out(y)

            if len(results) > 0 and len(results[0]) > 0:
                # Interpret output, only one frame is used
                det_label = results[0][:, 0]
                det_conf = results[0][:, 1]
                det_xmin = results[0][:, 2]
                det_ymin = results[0][:, 3]
                det_xmax = results[0][:, 4]
                det_ymax = results[0][:, 5]

                top_indices = [i for i, conf in enumerate(det_conf) if conf >= conf_thresh]

                top_conf = det_conf[top_indices]
                top_label_indices = det_label[top_indices].tolist()
                top_xmin = det_xmin[top_indices]
                top_ymin = det_ymin[top_indices]
                top_xmax = det_xmax[top_indices]
                top_ymax = det_ymax[top_indices]

                for i in range(top_conf.shape[0]):
                    if(top_conf[i] < 0.9):
                        continue
                    xmin = int(round(top_xmin[i] * to_draw.shape[1]))
                    ymin = int(round(top_ymin[i] * to_draw.shape[0]))
                    xmax = int(round(top_xmax[i] * to_draw.shape[1]))
                    ymax = int(round(top_ymax[i] * to_draw.shape[0]))

                    # Draw the box on top of the to_draw image
                    class_num = int(top_label_indices[i])
                    cv2.rectangle(to_draw, (xmin, ymin), (xmax, ymax),
                                  self.class_colors[class_num], 2)
                    text = self.class_names[class_num] + " " + ('%.2f' % top_conf[i])

                    print(text)

                    text_top = (xmin, ymin-10)
                    text_bot = (xmin + 80, ymin + 5)
                    text_pos = (xmin + 5, ymin)
                    cv2.rectangle(to_draw, text_top, text_bot, self.class_colors[class_num], -1)
                    cv2.putText(to_draw, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

            # Calculate FPS
            # This computes FPS for everything, not just the model's execution
            # which may or may not be what you want
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0

            # Draw FPS in top left corner
            cv2.rectangle(to_draw, (0,0), (50, 17), (255,255,255), -1)
            cv2.putText(to_draw, fps, (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1)

            cv2.imshow("SSD result", to_draw)
            cv2.waitKey(10)

            self.current_fps += 1
            self.exec_time = None

    def timer_callback(self):
        self.current_time += 1
        if self.exec_time:
            self.extra_time = time.time() - self.exec_time
        fps = self.calc_fps(self.current_fps, self.prev_extra_time, self.extra_time)
        self.fps_time_slot.append((self.current_time, fps))
        self.current_fps = 0
        self.prev_extra_time = self.extra_time
        if not self.is_finish:
            print("\rfps: {}, {} sec".format(fps, self.current_time), end="")
            self.restart_timer()

    def restart_timer(self):
        self.timer = Timer(1, self.timer_callback)
        self.timer.start()

    @staticmethod
    def calc_fps(current_fps, prev_extra_time, extra_time):
        if not prev_extra_time and not extra_time:
            return current_fps
        if not prev_extra_time:
            return round(current_fps / (1 - extra_time), 1)
        if not extra_time:
            return round(current_fps / (1 + prev_extra_time), 1)
        return round(current_fps / (1 - extra_time + prev_extra_time), 1)

    @staticmethod
    def calc_avg_fps(fps_time_slot):
        fps_list = [fps for sec, fps in fps_time_slot if fps != 0][1:-1]
        avg_fps = round(sum(fps_list) / len(fps_list), 1)
        total_time = len(fps_list)
        return avg_fps, total_time

    def draw_fps(self, fps_time_slot):
        x_range = [index for index, value in enumerate(fps_time_slot)]
        y_fps = [value[1] for index, value in enumerate(fps_time_slot)]
        plt.ylim((0, 20))
        plt.plot(x_range, y_fps, c='r')
        plt.xlabel('time')
        plt.ylabel('fps')
        plt.show()