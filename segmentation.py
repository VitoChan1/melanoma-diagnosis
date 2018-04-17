#!/usr/bin/python
import tensorflow as tf
import os
import argparse
# import numpy as np

H, W = (480, 480)
epochs = 100
batch_size = 4


def random_crop_and_pad_image_and_labels(image, labels, size):
    """Randomly crops `image` together with `labels`.

    Args:
      image: A Tensor with shape [D_1, ..., D_K, N]
      labels: A Tensor with shape [D_1, ..., D_K, M]
      size: A Tensor with shape [K] indicating the crop size.
    Returns:
      A tuple of (cropped_image, cropped_label).
    """
    combined = tf.concat([image, labels], axis=2)
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(
        combined, 0, 0,
        tf.maximum(size[0], image_shape[0]),
        tf.maximum(size[1], image_shape[1]))
    last_label_dim = tf.shape(labels)[-1]
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(
        combined_pad,
        size=tf.concat([size, [last_label_dim + last_image_dim]],
                       axis=0))
    return (combined_crop[:, :, :last_image_dim],
            combined_crop[:, :, last_image_dim:])


def read_batch(data_dir, label_dir, batch_size, size=[H, W]):
    with tf.name_scope('read'):
        # create input filenames queue
        data_filenames = tf.convert_to_tensor(
            sorted(os.listdir(data_dir)), dtype=tf.string)
        label_filenames = tf.convert_to_tensor(
            sorted(os.listdir(label_dir)), dtype=tf.string)
        input_queue = tf.train.slice_input_producer(
            [data_filenames, label_filenames], num_epochs=None, shuffle=True)

        # read data and label
        data = tf.read_file(data_dir + '/' + input_queue[0])
        data = tf.image.decode_jpeg(data, channels=3)
        label = tf.read_file(label_dir + '/' + input_queue[1])
        label = tf.image.decode_png(label, channels=1)

        # random crop
        data, label = random_crop_and_pad_image_and_labels(
            data, label, size=[H, W])
        data.set_shape([H, W, 3])
        label.set_shape([H, W, 1])
        # data = tf.image.resize_images(data, size)
        # label = tf.image.resize_images(label, size)

        # batching
        data_batch, label_batch = tf.train.batch(
            [data, label], batch_size=batch_size)

    return data_batch, label_batch


def preprocess(data, label):
    with tf.name_scope('preprocess'):
        px_mean = tf.constant([183.55, 157.87, 143.42],
                              tf.float32, name='px_mean')
        data = tf.subtract(tf.cast(data, tf.float32), px_mean)
        label = tf.one_hot(tf.cast(tf.reduce_sum(
            label, axis=-1), tf.int32) / 255, 2)
    return data, label


def weight_variable(shape):
    with tf.name_scope('truncated_normal_VAR'):
        initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def model(data):
    # load pretrained model
    saver = tf.train.import_meta_graph('./ResNet-L50.meta')
    var_list1 = tf.trainable_variables()

    # fuse pretrained model with our input
    graph = tf.get_default_graph()
    sub = graph.get_tensor_by_name('sub:0')
    tf.contrib.graph_editor.reroute_ts([data], [sub])

    # deconv processes
    res5 = graph.get_tensor_by_name('scale5/block3/Relu:0')
    w_conv1 = weight_variable([1, 1, 2048, 2])
    conv1 = tf.nn.conv2d(res5, w_conv1, strides=[
                         1, 1, 1, 1], padding='SAME')

    with tf.name_scope('upsample'):
        # w_deconv1 = weight_variable([4, 4, 2, 2])
        deconv1 = tf.layers.conv2d_transpose(
            conv1, 2, (4, 4), strides=(2, 2),
            padding='SAME', use_bias=False, name='deconv1', trainable=True)
        # deconv1 = tf.nn.conv2d_transpose(
        #    conv1, w_deconv1, output_shape=[batch_size, H / 16, W / 16, 2],
        #    strides=[1, 2, 2, 1], padding='SAME')
    w_deconv1 = tf.trainable_variables()[-1]

    with tf.name_scope('fuse_upsample1'):
        res4 = graph.get_tensor_by_name('scale4/block6/Relu:0')
        w_conv2 = weight_variable([1, 1, 1024, 2])
        conv2 = tf.nn.conv2d(res4, w_conv2, strides=[
                             1, 1, 1, 1], padding='SAME')
        fuse1 = deconv1 + conv2
        with tf.name_scope('upsample'):
            # w_deconv2 = weight_variable([4, 4, 2, 2])
            deconv2 = tf.layers.conv2d_transpose(
                fuse1, 2, (4, 4), strides=(2, 2),
                padding='SAME', use_bias=False, name='deconv2', trainable=True)
            # deconv2 = tf.nn.conv2d_transpose(
            #    fuse1, w_deconv2, output_shape=[batch_size, H / 8, W / 8, 2],
            #    strides=[1, 2, 2, 1], padding='SAME')
    w_deconv2 = tf.trainable_variables()[-1]

    with tf.name_scope('fuse_upsample2'):
        res3 = graph.get_tensor_by_name('scale3/block4/Relu:0')
        w_conv3 = weight_variable([1, 1, 512, 2])
        conv3 = tf.nn.conv2d(res3, w_conv3, strides=[
                             1, 1, 1, 1], padding='SAME')
        fuse2 = deconv2 + conv3
        with tf.name_scope('upsample'):
            # w_deconv3 = weight_variable([16, 16, 2, 2])
            deconv3 = tf.layers.conv2d_transpose(
                fuse2, 2, (16, 16), strides=(8, 8),
                padding='SAME', use_bias=False, name='deconv3', trainable=True)
            # deconv3 = tf.nn.conv2d_transpose(
            #    fuse2, w_deconv3, output_shape=[batch_size, H, W, 2],
            #    strides=[1, 8, 8, 1], padding='SAME')
    w_deconv3 = tf.trainable_variables()[-1]

    var_list1.extend([w_conv1, w_conv2, w_conv3])
    var_list2 = [w_deconv1, w_deconv2, w_deconv3]

    return deconv3, var_list1, var_list2, saver


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


def main(logdir='./logs/seg'):
    data_batch, label_batch = read_batch(
        './ISBI2016_ISIC_Part1_Training_Data_cropped',
        './ISBI2016_ISIC_Part1_Training_GroundTruth_cropped', batch_size)
    data, label = preprocess(data_batch, label_batch)

    result, var_list1, var_list2, pretrained_saver = model(data)

    with tf.name_scope('softmax_with_loss'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=result, labels=label, dim=3))

    train_op = optimize_with_two_lr(
        tf.train.AdamOptimizer, cross_entropy,
        var_list1, var_list2, 0.001, 0.0001)
    # train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    with tf.name_scope('evaluation'):
        prediction = tf.cast(tf.argmax(result, 3), tf.bool)
        ground_truth = tf.cast(tf.argmax(label, 3), tf.bool)
        TP = tf.reduce_sum(tf.cast(tf.logical_and(
            prediction, ground_truth), tf.int32), axis=[1, 2])
        TN = tf.reduce_sum(tf.cast(tf.logical_not(
            tf.logical_or(prediction, ground_truth)), tf.int32), axis=[1, 2])
        FP = tf.reduce_sum(tf.cast(tf.logical_and(
            prediction, tf.logical_not(ground_truth)), tf.int32), axis=[1, 2])
        FN = tf.reduce_sum(tf.cast(tf.logical_and(
            tf.logical_not(prediction), ground_truth), tf.int32), axis=[1, 2])

        # accuracy(AC), sensitivity(SE), specificity(SP)
        # Jaccard index(JA), Dice coefficient(DI)
        AC = tf.reduce_mean(tf.divide(TP + TN, TP + TN + FP + FN))
        SE = tf.reduce_mean(tf.divide(TP, TP + FN))
        SP = tf.reduce_mean(tf.divide(TN, TN + FP))
        JA = tf.reduce_mean(tf.divide(TP, TP + FN + FP))
        DI = tf.reduce_mean(tf.divide(2 * TP, 2 * TP + FN + FP))

    tf.summary.scalar('cross_entropy', cross_entropy)
    tf.summary.scalar('accuracy', AC)
    tf.summary.scalar('sensitivity', SE)
    tf.summary.scalar('specificity', SP)
    tf.summary.scalar('Jaccard index', JA)
    tf.summary.scalar('Dice coefficient', DI)
    merged_summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    saver = tf.train.Saver(max_to_keep=10, keep_checkpoint_every_n_hours=1)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        pretrained_saver.restore(sess, './ResNet-L50.ckpt')
        saver.save(sess, 'ckpts/seg')

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        acc = 0
        ja = 0
        loss = 0
        for i in range(epochs):
            for j in range(900 / batch_size):
                acc_step, ja_step, loss_step, summary_str, _ = sess.run(
                    [AC, JA, cross_entropy, merged_summary_op, train_op])
                summary_writer.add_summary(
                    summary_str, i * 900 / batch_size + j)
                acc += acc_step
                ja += ja_step
                loss += loss_step
                if j % 25 == 24:
                    acc /= 25.0
                    ja /= 25.0
                    loss /= 25.0
                    print 'epoch', i, 'batch', j
                    print 'accuracy', acc, 'Jaccard index', ja,\
                          'cross_entropy', loss
                    acc = 0
                    ja = 0
                    loss = 0
            saver.save(sess, 'ckpts/seg/seg', global_step=(i + 1)
                       * 900 / batch_size)

        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Fully convolutional network for dermoscopic \
                     image segmentation")
    ap.add_argument('-l', '--logdir', required=False, default='./logs/seg',
                    help="specify log dir, ./logs/seg by default")
    args = vars(ap.parse_args())
    logdir = args['logdir']
    main(logdir)
