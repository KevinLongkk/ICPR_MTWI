import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import config as cfg


def build_network(images,
                  num_outputs,
                  alpha,
                  keep_prob=1.0,
                  is_training=True,
                  scope='yolo'):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=leaky_relu(alpha),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            variables_collections='Variables'):
            net = slim.conv2d(images, 64, 3, 2, padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 128, 3, padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 256, 3, padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 256, 3, padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 512, 3, 2, padding='VALID', scope='conv_28')
            net = slim.conv2d(net, 1024, 3, scope='conv_29')
            net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
            net = slim.flatten(net, scope='flat_32')
            net = slim.fully_connected(net, 512, scope='fc_33')
            net = slim.fully_connected(net, 4096, scope='fc_34')
            net = slim.dropout(net, keep_prob=keep_prob,
                               is_training=is_training, scope='dropout_35')
            net = slim.fully_connected(net, num_outputs,
                                       activation_fn=None, scope='fc_36')
        return net



def _build_network(
        images,
        num_outputs,
        alpha,
        keep_prob=1.0,
        is_training=True,
        scope='yolo'):
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            activation_fn=leaky_relu(alpha),
                            weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                            weights_regularizer=slim.l2_regularizer((0.0005)),
                            variables_collections='Variables'):
            net = tf.pad(images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]), name='pad_1')
            net = slim.conv2d(net, 64, 7, 2, padding='VALID', scope='conv_2', trainable=False)
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
            net = slim.conv2d(net, 192, 3, scope='conv_4', trainable=False)
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
            net = slim.conv2d(net, 128, 1, scope='conv_6')
            net = slim.conv2d(net, 256, 3, scope='conv_7')
            net = slim.conv2d(net, 256, 1, scope='conv_8')
            net = slim.conv2d(net, 512, 3, scope='conv_9')
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
            net = slim.conv2d(net, 256, 1, scope='conv_11')
            net = slim.conv2d(net, 512, 3, scope='conv_12')
            net = slim.conv2d(net, 256, 1, scope='conv_13')
            net = slim.conv2d(net, 512, 3, scope='conv_14')
            net = slim.conv2d(net, 256, 1, scope='conv_15')
            net = slim.conv2d(net, 512, 3, scope='conv_16')
            net = slim.conv2d(net, 256, 1, scope='conv_17')
            net = slim.conv2d(net, 512, 3, scope='conv_18')
            net = slim.conv2d(net, 512, 1, scope='conv_19')
            # tf.summary.histogram('conv19', net)
            net = slim.conv2d(net, 1024, 3, scope='conv_20')
            # tf.summary.histogram('conv20', net)
            net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
            net = slim.conv2d(net, 512, 1, scope='conv_22')
            net = slim.conv2d(net, 1024, 3, scope='conv_23')
            net = slim.conv2d(net, 512, 1, scope='conv_24')
            net = slim.conv2d(net, 1024, 3, scope='conv_25')
            net = slim.conv2d(net, 1024, 3, scope='conv_26')
            # tf.summary.histogram('conv26', net)
            net = tf.pad(net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]), name='pad_27')
            net = slim.conv2d(net, 1024, 3, 2, padding='VALID', scope='conv_28')
            net = slim.conv2d(net, 1024, 3, scope='conv_29')
            net = slim.conv2d(net, 1024, 3, scope='conv_30')
            net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
            net = slim.flatten(net, scope='flat_32')
            net = slim.fully_connected(net, 512, scope='fc_33')
            net = slim.fully_connected(net, 4096, scope='fc_34')
            net = slim.dropout(net, keep_prob=keep_prob,
                               is_training=is_training, scope='dropout_35')
            net = slim.fully_connected(net, num_outputs,
                                       activation_fn=None, scope='fc_36')
            # net ~ batch * 7 * 7 * 30
        return net



def loss_layer(predicts, labels, scope='loss_layer'):
    with tf.variable_scope(scope):
        pre_response = tf.reshape(predicts[:, :cfg.CELL_SIZE*cfg.CELL_SIZE],
                                  [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 1])
        pre_boxes = tf.reshape(predicts[:, cfg.CELL_SIZE*cfg.CELL_SIZE:],
                               [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 8])

        response = tf.reshape(labels[:, :, :, 0], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 1])
        boxes = tf.reshape(labels[:, :, :, 1:], [cfg.BATCH_SIZE, cfg.CELL_SIZE, cfg.CELL_SIZE, 8])

        # pre_response = tf.clip_by_value(pre_response, 1e-5, 1.0)
        # obj_delta = -(response * tf.log(pre_response) + (1 - response) * tf.log(1 - pre_response))
        # obj_loss = tf.reduce_mean(tf.reduce_sum(tf.square(response - pre_response), axis=[1, 2, 3])) * 3
        # obj_loss = tf.reduce_mean(tf.reduce_sum(obj_delta, axis=[1, 2, 3])) * 5.0
        obj_loss = tf.reduce_sum(tf.square(pre_response - response))

        no_obj_mask = 1 - response
        boxes_delta = tf.square(tf.sqrt(tf.abs(boxes)) - tf.sqrt(tf.abs(pre_boxes))) * response * 3.0
        no_obj_boxes_delta = tf.square(tf.sqrt(tf.abs(boxes)) - tf.sqrt(tf.abs(pre_boxes))) * no_obj_mask * 1.0
        coord_loss = tf.reduce_mean(tf.reduce_sum(boxes_delta, axis=[1, 2, 3]) +
                                    tf.reduce_sum(no_obj_boxes_delta, axis=[1, 2, 3]))
        # coord_loss = tf.reduce_sum(boxes_delta)

        tf.losses.add_loss(obj_loss)
        tf.losses.add_loss(coord_loss)

        tf.summary.scalar('obj_loss', obj_loss)
        tf.summary.scalar('coord_loss', coord_loss)
        tf.summary.scalar('total_loss', obj_loss+coord_loss)

    return tf.losses.get_total_loss(), boxes, pre_boxes


def leaky_relu(alpha):
    def op(inputs):
        return tf.maximum(alpha * inputs, inputs, name='leaky_relu')
    return op