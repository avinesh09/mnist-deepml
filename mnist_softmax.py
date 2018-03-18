# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter

FLAGS = None

EXPORT_PATH = '/opt/model/'


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


VERSION = 1


def main(_):
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    variable_summaries(W)
    variable_summaries(b)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    # The raw formulation of cross-entropy,
    #
    #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
    #                                 reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    tf.summary.scalar('cross_entropy', cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    pred = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    sess = tf.InteractiveSession()
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(EXPORT_PATH + 'log' + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(EXPORT_PATH + 'log' + '/test')
    tf.global_variables_initializer().run()
    # Train
    for i in range(50000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        if i % 10 == 0:  # Record summaries and test-set accuracy
            summary, acc = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
            test_writer.add_summary(summary, i)
        else:  # Record train set summaries, and train
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_step],
                                      feed_dict={x: batch_xs, y_: batch_ys},
                                      options=run_options,
                                      run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run([merged, train_step], feed_dict={x: batch_xs, y_: batch_ys})
                train_writer.add_summary(summary, i)

    # Test trained model
    pred = tf.argmax(y, axis=1)
    correct_prediction = tf.equal(pred, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                        y_: mnist.test.labels}))

    saver = tf.train.Saver(sharded=True)
    model_exporter = exporter.Exporter(saver)
    model_exporter.init(
        sess.graph.as_graph_def(),
        named_graph_signatures={
            'inputs': exporter.generic_signature({'x': x}),
            'outputs': exporter.generic_signature({'pred': pred})})
    model_exporter.export(EXPORT_PATH, tf.constant(VERSION), sess)
    train_writer.close()
    test_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
