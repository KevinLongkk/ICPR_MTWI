import os

BATCH_SIZE = 32

IMAGE_SIZE = 448

CELL_SIZE = 19

DATA_PATH = r'/home/kevin/DataSet/tianchi/ICPR_MTWI'

DATA_PART1_PATH = os.path.join(DATA_PATH, 'ICPR_text_train_part1_20180211')

DATA_PART2_PATH = os.path.join(DATA_PATH, 'ICPR_text_train_part2_20180313')

OUTPUT_PATH = 'output'

WEIGHTS_DIR = 'weights/YOLO_small.ckpt'


ALPHA = 0.0001




LEARNING_RATE = 0.0001

DECAY_STEPS = 30000

DECAY_RATE = 0.1