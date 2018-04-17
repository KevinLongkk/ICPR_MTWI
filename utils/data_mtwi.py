import config as cfg
import os
import numpy as np
import cv2



def get_data(data_size, batch_size=cfg.BATCH_SIZE):

    images = np.zeros((batch_size, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3), np.uint8)
    labels = np.zeros((batch_size, cfg.CELL_SIZE, cfg.CELL_SIZE, 9))
    if data_size == 1000:
        DATA_PART_PATH = cfg.DATA_PART1_PATH
    elif data_size == 9000:
        DATA_PART_PATH = cfg.DATA_PART2_PATH
    else:
        return
    image_path = os.path.join(DATA_PART_PATH, 'image_' + str(data_size))
    label_path = os.path.join(DATA_PART_PATH, 'txt_' + str(data_size))
    image_list = os.listdir(image_path)
    image_list = np.array(image_list)
    randint = np.random.randint(0, data_size, batch_size)
    index = 0
    for i in randint:
        try:
            image = cv2.imread(os.path.join(image_path, image_list[i]))
            height, width, _ = image.shape
            image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[index, :, :, :] = (image / 255.0) * 2.0 - 1.0

            w_ratio = cfg.IMAGE_SIZE / width
            h_ratio = cfg.IMAGE_SIZE / height

            txt_path = os.path.join(label_path, image_list[i][:-3]+'txt')
            with open(txt_path) as f:
                lines = f.readlines()
                for line in lines:
                    c = line.split(',')
                    c = list(map(float, c[:-1]))
                    c_w, c_h = 0, 0
                    for k in [0, 2, 4, 6]:
                        c[k] *= w_ratio
                        c_w += c[k]
                    for k in [1, 3, 5, 7]:
                        c[k] *= h_ratio
                        c_h += c[k]

                    c_w = c_w / 4
                    c_h = c_h / 4

                    c_x = int(c_w / cfg.IMAGE_SIZE * cfg.CELL_SIZE)
                    c_y = int(c_h / cfg.IMAGE_SIZE * cfg.CELL_SIZE)

                    cell_center_x = (c_x + 0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE
                    cell_center_y = (c_y + 0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE

                    for k in [0, 2, 4, 6]:
                        c[k] -= cell_center_x
                    for k in [1, 3, 5, 7]:
                        c[k] -= cell_center_y

                    # c = list(map(int, c))
                    # pts = np.array(c)
                    # pts = pts.reshape(-1, 4, 2)
                    # cv2.polylines(images[index], pts, True, (0, 0, 255))
                    # cv2.imshow('tt', images[index])
                    # cv2.waitKey()
                    labels[index, c_y, c_x, 0] = 1
                    labels[index, c_y, c_x, 1:] = c
            index += 1
        except:
            # print(image_list[i])
            pass
    return images, labels


def test_get_data(data_size, batch_size=cfg.BATCH_SIZE):

    images = np.zeros((batch_size, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3))
    labels = np.zeros((batch_size, cfg.CELL_SIZE, cfg.CELL_SIZE, 9))
    if data_size == 1000:
        DATA_PART_PATH = cfg.DATA_PART1_PATH
    elif data_size == 9000:
        DATA_PART_PATH = cfg.DATA_PART2_PATH
    else:
        return
    image_path = os.path.join(DATA_PART_PATH, 'image_' + str(data_size))
    label_path = os.path.join(DATA_PART_PATH, 'txt_' + str(data_size))
    image_list = os.listdir(image_path)
    image_list = np.array(image_list)
    index = 0
    randint = [0]
    for i in randint:
        try:
            image = cv2.imread(os.path.join(image_path, image_list[i]))
            height, width, _ = image.shape
            image = cv2.resize(image, (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            images[index, :, :, :] = (image / 255.0) * 2.0 - 1.0

            w_ratio = cfg.IMAGE_SIZE / width
            h_ratio = cfg.IMAGE_SIZE / height

            txt_path = os.path.join(label_path, image_list[i][:-3] + 'txt')
            with open(txt_path) as f:
                lines = f.readlines()
                for line in lines:
                    c = line.split(',')
                    c = list(map(float, c[:-1]))
                    c_w, c_h = 0, 0
                    for k in [0, 2, 4, 6]:
                        c[k] *= w_ratio
                        c_w += c[k]
                    for k in [1, 3, 5, 7]:
                        c[k] *= h_ratio
                        c_h += c[k]

                    c_w = c_w / 4
                    c_h = c_h / 4

                    c_x = int(c_w / cfg.IMAGE_SIZE * cfg.CELL_SIZE)
                    c_y = int(c_h / cfg.IMAGE_SIZE * cfg.CELL_SIZE)

                    cell_center_x = (c_x + 0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE
                    cell_center_y = (c_y + 0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE

                    for k in [0, 2, 4, 6]:
                        c[k] -= cell_center_x
                    for k in [1, 3, 5, 7]:
                        c[k] -= cell_center_y

                    # c = list(map(int, c))
                    # pts = np.array(c)
                    # pts = pts.reshape(-1, 4, 2)
                    # cv2.polylines(images[index], pts, True, (0, 0, 255))
                    # cv2.imshow('tt', images[index])
                    # cv2.waitKey()
                    labels[index, c_y, c_x, 0] = 1
                    labels[index, c_y, c_x, 1:] = c
            index += 1
        except:
            # print(image_list[i])
            pass
    return images, labels

# get_data(9000)

# for i in range(10):
#     images, labels = get_data(9000, 128)
#     print(images.shape)
#     print(labels.shape)
#     image = np.array((images[0] + 1.0) / 2.0 * 255.0).astype(np.uint8)
#     label = labels[0]
#     for i in range(cfg.CELL_SIZE):
#         for j in range(cfg.CELL_SIZE):
#             if label[i, j, 0] == 1:
#                 c = label[i, j, 1:]
#                 c = list(map(int, c))
#                 pts = np.array(c)
#                 pts = pts.reshape(-1, 4, 2)
#                 cv2.polylines(image, pts, True, (0, 100, 255), 2)
#     cv2.imshow('tt', image)
#     cv2.waitKey()
