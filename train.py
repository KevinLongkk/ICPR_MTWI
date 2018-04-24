import tensorflow as tf
from utils import data_mtwi
import config as cfg
import datetime
import os
from net import yolo_net
from tensorflow.contrib import slim

import math, sys


def restart_program():
    """Restarts the current program.
    Note: this function does not return. Any cleanup action (like
    saving data) must be done before calling this function."""
    python = sys.executable
    os.execl(python, python, * sys.argv)


def train():

    output_dir = os.path.join(cfg.OUTPUT_PATH, datetime.datetime.now().strftime('%Y_%m_%d'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    Y = tf.placeholder(tf.float32, [None, cfg.CELL_SIZE, cfg.CELL_SIZE, 9])

    tf.summary.image('input', X, 10)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE, global_step, cfg.DECAY_STEPS,
                                               cfg.DECAY_RATE, True, name='learning_rate')

    logits = yolo_net.build_network(X, cfg.CELL_SIZE * cfg.CELL_SIZE * 9, cfg.ALPHA, True)
    loss, boxes, pre_boxes = yolo_net.loss_layer(logits, Y)

    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)
    optimizer_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)

    # tf.summary.scalar('learning_rate', learning_rate)
    #
    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # params = tf.trainable_variables()
    # gradients = tf.gradients(loss, params)
    #
    # clipped_gradients, norm = tf.clip_by_global_norm(gradients, 100)
    # optimizer_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    average_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([optimizer_op]):
        train_op = tf.group(average_op)

    saver = tf.train.Saver()
    # restorer = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        try:
            last_cke_path = tf.train.latest_checkpoint(cfg.OUTPUT_PATH)
            saver.restore(sess, last_cke_path)
            # sess.run(tf.global_variables_initializer())
            # include = ['yolo/pad_1', 'yolo/conv_2', 'yolo/conv_4', 'yolo/conv_6', 'yolo/conv_7', 'yolo/conv_8', 'yolo/conv_9',
            #            'yolo/conv_11', 'yolo/conv_12', 'yolo/conv_13', 'yolo/conv_14', 'yolo/conv_15', 'yolo/conv_16', 'yolo/conv_17',
            #            'yolo/conv_18', 'yolo/conv_19']
            # exclude = ['fc_33', 'fc_34', 'fc_36', 'global_step']
            # variables_to_restore = slim.get_variables_to_restore(include=include)
            # restorer = tf.train.Saver(variables_to_restore)
            # restorer.restore(sess, cfg.WEIGHTS_DIR)
        except:
            print('Failed to restore checkpoint.Initializing variable instead.')
            sess.run(tf.global_variables_initializer())

        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter('log', sess.graph)

        for step in range(1, 300000):
            images, labels = data_mtwi.get_data(9000)
            feed_dict = {X: images, Y: labels}
            losses, learning_rate_, _ = sess.run([loss, learning_rate, train_op], feed_dict=feed_dict)
            #当出现NAN时,程序重新启动
            if math.isnan(losses):
                print('restart')
                restart_program()
            if step % 10 == 0:
                # print(sess.run(boxes, feed_dict=feed_dict))
                # print(sess.run(pre_boxes, feed_dict=feed_dict))
                # print(sess.run(gradients, feed_dict=feed_dict))
                global_step_ = sess.run(global_step)
                summary_str = sess.run(merged_summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, global_step_)
                print('{}steps, total_loss is {}, learning_rate is {}'.format(global_step_, losses, learning_rate_))
            if step % 200 == 0:
                saver.save(sess, save_path=output_dir, global_step=global_step)
                print('save model success')


if __name__ == '__main__':
    train()
