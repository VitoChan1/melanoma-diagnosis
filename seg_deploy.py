import tensorflow as tf
from scipy import misc
import argparse


def main(input_path, output_path):
    input = tf.placeholder(tf.uint8)
    input_float = tf.cast(input, tf.float32)
    saver = tf.train.import_meta_graph('./ckpts/seg.meta')
    # fuse trained model with our input
    graph = tf.get_default_graph()
    read = graph.get_tensor_by_name('preprocess/Cast:0')
    tf.contrib.graph_editor.reroute_ts([input_float], [read])

    # convert and reshape from (1, H, W, 2) to (H, W)
    result = graph.get_tensor_by_name(
        'fuse_upsample2/upsample/deconv3/conv2d_transpose:0')
    result = tf.reduce_sum(result, axis=0)
    result = tf.cast(tf.argmax(result, -1) * 255, tf.uint8)

    img = misc.imread(input_path)
    with tf.Session() as sess:
        tf.train.import_meta_graph('./ckpts/seg-675.meta')
        saver.restore(sess, "./ckpts/seg-675")
        print tf.train.latest_checkpoint('./ckpts')
        output = sess.run(result, feed_dict={input: [img]})
        print output.shape
        misc.imsave(output_path, output)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description="Fully convolutional network for dermoscopic image segmentation")
    ap.add_argument('-i', '--input', required=True,
                    help="specify input image for segmentation")
    ap.add_argument('-o', '--output', required=False, default='./out.png',
                    help="specify output path")
    args = vars(ap.parse_args())
    input = args['input']
    output = args['output']
    main(input, output)
