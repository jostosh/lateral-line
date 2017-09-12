import os
import tensorflow as tf


def add_summary(summary):
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/train', summary)
    tf.add_to_collection(tf.GraphKeys.SUMMARIES + '/test', summary)


def init_summary_writer(logdir, sess, sub):
    writer = tf.summary.FileWriter(os.path.join(logdir, sub), sess.graph)
    summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES + '/' + sub))
    return summary_op, writer