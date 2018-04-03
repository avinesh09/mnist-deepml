# Copyright 2016 Google Inc. All Rights Reserved.
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

#!/usr/bin/env python2.7
r"""Train and export a simple Softmax Regression TensorFlow model.
The model is from the TensorFlow "MNIST For ML Beginner" tutorial. This program
simply follows all its training instructions, and uses TensorFlow SavedModel to
export the trained model with proper signatures that can be loaded by standard
tensorflow_model_server.
Usage: mnist_saved_model.py [--training_iteration=x] [--model_version=y] \
    export_dir
"""

from __future__ import print_function

import os
import sys

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.contrib import predictor


import mnist_input_data

tf.app.flags.DEFINE_integer('training_iteration', 10000,
                            'number of training iterations.')
tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('data_dir', '/tmp/model/mnist/data', 'Working directory.')
tf.app.flags.DEFINE_string('model_dir', '/opt/mnist/model', 'export model directory.')
tf.app.flags.DEFINE_string('summary_dir', '/opt/mnist/summaries', 'summaries directory.')
FLAGS = tf.app.flags.FLAGS


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

def main(_):
  '''
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: mnist_export.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
      print('Please specify a positive value for version number.')
      sys.exit(-1)
  '''

  #FLAGS.model_version = 4
  #FLAGS.model_dir = "model"
  #FLAGS.data_dir = "model/data"
  #FLAGS.summary_dir = "model/summ"
  MODEL_EXPORT_PATH = FLAGS.model_dir
  MODEL_SUMMARY_DIR = FLAGS.summary_dir

  # Train model
  print('Training model...')
  mnist = mnist_input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, 10])
  w = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  variable_summaries(w)
  variable_summaries(b)
  sess.run(tf.global_variables_initializer())
  y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, 10)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(
      tf.constant([str(i) for i in xrange(10)]))
  prediction_classes = table.lookup(tf.to_int64(indices))

  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(MODEL_SUMMARY_DIR + '/log' + '/train', sess.graph)
  test_writer = tf.summary.FileWriter(MODEL_SUMMARY_DIR + '/log' + '/test')

  for _ in range(FLAGS.training_iteration):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  summaries, _, train_accuracy = sess.run([merged, train_step, accuracy],
           feed_dict={
               x: mnist.test.images,
               y_: mnist.test.labels
           })
  print('training accuracy %g' % train_accuracy)
  print('Done training!')

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
  # export_path_base = sys.argv[-1]
  export_path = os.path.join(
      tf.compat.as_bytes(MODEL_EXPORT_PATH),
      tf.compat.as_bytes(str(FLAGS.model_version)))
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  classification_inputs = tf.saved_model.utils.build_tensor_info(
      serialized_tf_example)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
      prediction_classes)
  classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classification_inputs
          },
          outputs={
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
          },
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_x},
          outputs={'scores': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

  builder.save()

  print('Done exporting!')


def load(session, tag_constants, export_dir):
    return tf.saved_model.loader.load(session, tag_constants, export_dir)

if __name__ == '__main__':
    main(sys.argv)

if __name__ == '__mainold__':

    with tf.Session(graph=tf.Graph()) as sess:
        '''
        serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
        feature_configs = {'x': tf.FixedLenFeature(shape=[784], dtype=tf.float32), }
        tf_example = tf.parse_example(serialized_tf_example, feature_configs)
        x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
        y_ = tf.placeholder('float', shape=[None, 10])
        w = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x, w) + b, name='y')

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        load(sess, [tf.saved_model.tag_constants.SERVING], "model/1")

        test_data_set = mnist_input_data.read_data_sets("model/data").test

        print (len(test_data_set.images))
        print(len(test_data_set.labels))

        predictor = tf.argmax(y, 1)

        acc, predictions = sess.run([accuracy, predictor], feed_dict={
            x: test_data_set.images
        })

        print('Accuracy', acc)
        print('Predictions', predictions)
        '''
        from tensorflow.core.framework import types_pb2, tensor_shape_pb2
        test_data_set = mnist_input_data.read_data_sets("model/data").test
        inputs = {'x': tf.TensorInfo(
                        name='x:0',
                        dtype=types_pb2.DT_FLOAT,
                        tensor_shape=tensor_shape_pb2.TensorShapeProto())}
        outputs = {'y': tf.TensorInfo(
                        name='y:0',
                        dtype=types_pb2.DT_FLOAT)}
        signature_def = tf.saved_model.signature_def_utils.build_signature_def(
            inputs=inputs,
            outputs=outputs,
            method_name='tensorflow/serving/predict')


        saved_model_predictor = predictor.from_saved_model(export_dir="model/4", signature_def=signature_def)

        #print ("test data : ", test_data_set.images[0:5])
        output_dict = saved_model_predictor({'x': test_data_set.images[0:20]})

        from matplotlib import pyplot as plt
        import numpy as np

        def input_image(arr):
            two_d = (np.reshape(arr, (28, 28)) * 255).astype(np.uint8)
            plt.imshow(two_d, interpolation='nearest')
            return plt

        def output_image(arr):
            plt.imshow(arr, interpolation='nearest')
            return plt

        out_arr =  output_dict["y"]
        for i in range(10):
            input_image(test_data_set.images[i]).show()
            arr = [out_arr[i]]
            output_image(arr).show()

        print(output_dict)
        output_tensor_name = saved_model_predictor

        print(len(output_dict["y"]))
