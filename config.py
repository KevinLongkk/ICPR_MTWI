import os

BATCH_SIZE = 8

IMAGE_SIZE = 448

CELL_SIZE = 7

DATA_PATH = r'/home/kevin/DataSet/tianchi/ICPR_MTWI'

DATA_PART1_PATH = os.path.join(DATA_PATH, 'ICPR_text_train_part1_20180211')

DATA_PART2_PATH = os.path.join(DATA_PATH, 'ICPR_text_train_part2_20180313')

OUTPUT_PATH = 'output'


ALPHA = 0.1



LEARNING_RATE = 0.00001

DECAY_STEPS = 30000

DECAY_RATE = 0.1