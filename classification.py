#!/usr/bin/python
import tensorflow as tf
import os
import argparse
# import numpy as np

H, W = (250, 250)
epochs = 100
batch_size = 4


def read_batch(data_dir, label_csv, batch_size, size=[H, W]):
    with tf.name_scope('read'):
        # create input filenames queue
        data_filenames = tf.convert_to_tensor(
            sorted(os.listdir(data_dir)), dtype=tf.string)
        ground_truth_file = open(label_csv)
        labels = ground_truth_file.readlines()
        labels = [0 if l.strip().split(',')[1] ==
                  'benign' else 1 for l in labels]
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        input_queue = tf.train.slice_input_producer(
            [data_filenames, labels], num_epochs=None, shuffle=True)

        # read data image
        data = tf.read_file(data_dir + '/' + input_queue[0])
        data = tf.image.decode_jpeg(data, channels=3)

        # resize
        data = tf.image.resize_images(data, size)
        data.set_shape([H, W, 3])

        # batching
        data_batch, label_batch = tf.train.batch(
            [data, input_queue[1]], batch_size=batch_size)

    return data_batch, label_batch


def preprocess(data, label):
    with tf.name_scope('preprocess'):
        px_mean = tf.constant([183.55, 157.87, 143.42],
                              tf.float32, name='px_mean')
        data = tf.subtract(tf.cast(data, tf.float32), px_mean)
        label = tf.one_hot(label, 2)
    return data, label


def weight_variable(shape):
    with tf.name_scope('truncated_normal_VAR'):
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def model(data):
    # load pretrained model
    saver = tf.train.import_meta_graph('./ResNet-L50.meta')
    # var_list1 = tf.trainable_variables()

    # fuse pretrained model with our input
    graph = tf.get_default_graph()
    sub = graph.get_tensor_by_name('sub:0')
    tf.contrib.graph_editor.reroute_ts([data], [sub])

    # fully connected layer
    with tf.name_scope('fc2'):
        avg_pool = graph.get_tensor_by_name('avg_pool:0')
        keep_prob = tf.placeholder(tf.float32)
        avg_pool = tf.nn.dropout(avg_pool, keep_prob)
        W_fc = weight_variable([2048, 2])
        b_fc = bias_variable([2])
        fc = tf.matmul(avg_pool, W_fc) + b_fc

    return fc, saver, keep_prob


def optimize_with_two_lr(optimizer, loss, var_list1, var_list2, lr1, lr2):
    opt1 = optimizer(lr1)
    opt2 = optimizer(lr2)
    grads = tf.gradients(loss, var_list1 + var_list2)
    grads1 = grads[:len(var_list1)]
    grads2 = grads[len(var_list1):]
    train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
    train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
    train_op = tf.group(train_op1, train_op2)
    return train_op


def main(logdir='./logs/cla'):
    data_batch, label_batch = read_batch(
        './ISBI2016_ISIC_Part3B_Training_Data_tight_cropped',
        './ISBI2016_ISIC_Part3B_Training_GroundTruth.csv', batch_size)
    data, label = preprocess(data_batch, label_batch)

    result, pretrained_saver, keep_prob = model(data)

    with tf.name_scope('softmax_with_loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=result, labels=label, dim=1))

    # train_op = optimize_with_two_lr(
    #    tf.train.AdamOptimizer, cross_entropy,
    #    var_list1, var_list2, 0.001, 0.0001)
    train_op = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    with tf.name_scope('evaluation'):
        prediction = tf.cast(tf.argmax(result, 1), tf.bool)
        ground_truth = tf.cast(tf.argmax(label, 1), tf.bool)
        TP = tf.reduce_sum(tf.cast(tf.logical_and(
            prediction, ground_truth), tf.int32))
        TN = tf.reduce_sum(tf.cast(tf.logical_not(
            tf.logical_or(prediction, ground_truth)), tf.int32))
        FP = tf.reduce_sum(tf.cast(tf.logical_and(
            prediction, tf.logical_not(ground_truth)), tf.int32))
        FN = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.logical_not(prediction), ground_truth), tf.int32))

    summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pretrained_saver.restore(sess, './ResNet-L50.ckpt')
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        loss = 0
        for i in range(epochs):
            for j in range(900 / batch_size):
                TP_step, TN_step, FP_step, FN_step, loss_step, _ = sess.run(
                    [TP, TN, FP, FN, cross_entropy, train_op],
                    feed_dict={keep_prob: 0.5})
                tp += TP_step
                tn += TN_step
                fp += FP_step
                fn += FN_step
                loss += loss_step

                if j % 25 == 24:
                    acc = (tp + tn) / float(tp + tn + fp + fn)
                    se = tp / float(tp + fn)
                    sp = tn / float(tn + fp)
                    loss /= 25.0
                    my_summary = tf.Summary(value=[
                        tf.Summary.Value(tag="accuracy", simple_value=acc),
                        tf.Summary.Value(tag="sensitivity", simple_value=se),
                        tf.Summary.Value(tag="specificity", simple_value=sp),
                        tf.Summary.Value(tag="loss", simple_value=loss),
                    ])
                    summary_writer.add_summary(
                        my_summary, i * 900 / batch_size + j)
                    print 'epoch', i + 1, 'batch', j + 1
                    print 'accuracy', acc, 'cross_entropy', loss
                    acc = 0
                    loss = 0
            saver.save(sess, 'ckpts/cla/cla', global_step=(i + 1)
                       * 900 / batch_size)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Deep residual network for dermoscopic \
                     image classification")
    ap.add_argument('-l', '--logdir', required=False, default='./logs/cla',
                    help="specify log dir, ./logs/cla by default")
    args = vars(ap.parse_args())
    logdir = args['logdir']
    main(logdir)
