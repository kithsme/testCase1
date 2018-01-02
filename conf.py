import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

class conf():

    def __init__(self):
        # config for preprocessing and image processing
        self.MIN_LONG = 127.07
        self.MAX_LONG = 127.14
        self.MIN_LAT = 37.48
        self.MAX_LAT = 37.53
        self.GRAY_SCALE = 1
        self.STEP = 32
        self.TIME_WINDOW = 5 # minutes
        self.DRAW_FIGURE = True
        self.SEED = 1120

        # config for training
        self.BATCH_SIZE = 50
        self.CONV1_FILTER_WIDTH = 5
        self.CONV1_FILTER_HEIGHT = 5
        self.CONV1_OUT_CH_NUM = 32
        self.CONV2_FILTER_WIDTH = 5
        self.CONV2_FILTER_HEIGHT = 5
        self.CONV2_OUT_CH_NUM = 64
        self.CONV3_FILTER_WIDTH = 3
        self.CONV3_FILTER_HEIGHT = 3
        self.CONV3_OUT_CH_NUM = 128
        self.CONV4_FILTER_WIDTH = 3
        self.CONV4_FILTER_HEIGHT = 3
        self.CONV4_OUT_CH_NUM = 256
        self.FC_FILTER_WIDTH = 2
        self.FC_FILTER_HEIGHT = 2
        self.FULLY_CONNECTED_NUM = 1024
        self.DROP_OUT_PROB = 0.5
        self.TRANING_SET_RATE = 0.7
        self.ITERATION = 100
        self.CONFIG_ID = id_generator()
    

