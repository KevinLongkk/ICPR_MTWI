from net import yolo_net
import tensorflow as tf
import config as cfg
import cv2
import numpy as np


test_image = 'test/train_1.jpg'
raw_image = cv2.imread(test_image)
raw_image = cv2.resize(raw_image, (448, 448))
image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
image = (image / 255.0) * 2.0 - 1.0

X = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])

predict = yolo_net.build_network(X, cfg.CELL_SIZE*cfg.CELL_SIZE*9, cfg.ALPHA, keep_prob=1.0)
pre_response = tf.reshape(predict[:, :cfg.CELL_SIZE*cfg.CELL_SIZE],
                                  [1, cfg.CELL_SIZE, cfg.CELL_SIZE, 1])
pre_boxes = tf.reshape(predict[:, cfg.CELL_SIZE*cfg.CELL_SIZE:],
                               [1, cfg.CELL_SIZE, cfg.CELL_SIZE, 8])

saver = tf.train.Saver()
with tf.Session() as sess:
    try:
        last_cke_path = tf.train.latest_checkpoint(cfg.OUTPUT_PATH)
        saver.restore(sess, last_cke_path)
    except:
        print('Failed to restore checkpoint.')
        exit()
    predict_response, predict_boxes = sess.run([pre_response, pre_boxes], feed_dict={X: [image]})
    for i in range(cfg.CELL_SIZE):
        for j in range(cfg.CELL_SIZE):
            print(predict_response[0][i][j][0])
            if predict_response[0][i][j][0] > 0.1:
                pts = predict_boxes[0][i][j][:]
                print(pts)
                for index in [0, 2, 4, 6]:
                    pts[index] += ((i+0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE)
                for index in [1, 3, 5, 7]:
                    pts[index] += ((j+0.5) * cfg.IMAGE_SIZE / cfg.CELL_SIZE)
                pts = np.reshape(predict_boxes[0][i][j][:], [-1, 4, 2]).astype(np.int)
                print(pts)
                cv2.polylines(raw_image, pts, True, (0, 0, 255), 1)

cv2.imshow(' ', raw_image)
cv2.waitKey()

