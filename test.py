from net import yolo_net
import tensorflow as tf
import config as cfg
import cv2
import numpy as np


test_image = 'test/1.jpg'
image = cv2.imread(test_image)
image = cv2.resize(image, (448, 448))
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
            if predict_response[0][i][j][0] > 0:
                print(predict_boxes[0][i][j][:])
                pts = np.reshape(predict_boxes[0][i][j][:], [-1, 4, 2]).astype(np.int)
                cv2.polylines(image, pts, True, (0, 0, 255), 2)

cv2.imshow(' ', np.array((image + 1.0) / 2.0 * 255).astype(np.uint8))
cv2.waitKey()

