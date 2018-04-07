import tensorflow as tf
from utils import data_mtwi
import config as cfg
import datetime
import os
from net import yolo_net


def train():

    output_dir = os.path.join(cfg.OUTPUT_PATH, datetime.datetime.now().strftime('%Y_%m_%d'))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    X = tf.placeholder(tf.float32, [None, cfg.IMAGE_SIZE, cfg.IMAGE_SIZE, 3])
    Y = tf.placeholder(tf.float32, [None, cfg.CELL_SIZE, cfg.CELL_SIZE, 9])

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(cfg.LEARNING_RATE, global_step, cfg.DECAY_STEPS,
                                               cfg.DECAY_RATE, True, name='learning_rate')

    logits = yolo_net.build_network(X, cfg.CELL_SIZE * cfg.CELL_SIZE * 9, cfg.ALPHA, True)
    loss = yolo_net.loss_layer(logits, Y)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.9999)
    average_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([optimizer]):
        train_op = tf.group(average_op)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            last_cke_path = tf.train.latest_checkpoint(cfg.OUTPUT_PATH)
            saver.restore(sess, last_cke_path)
        except:
            print('Failed to restore checkpoint.Initializing variable instead.')
            sess.run(tf.global_variables_initializer())
        for step in range(1, 30000):
            images, labels = data_mtwi.test_get_data(9000)
            feed_dict = {X: images, Y: labels}
            losses, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            print('{}steps, total_loss is {}'.format(step, losses))
            if step % 100 == 0:
                saver.save(sess, save_path=output_dir, global_step=global_step)
                print('save model success')


if __name__ == '__main__':
    train()