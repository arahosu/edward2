# coding=utf-8
# Copyright 2019 The Edward2 Authors.
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

"""Wide ResNet 28-10 on CIFAR-10 trained with maximum likelihood.

Hyperparameters such as learning rate schedule differ slightly from
the original paper's code
(https://github.com/szagoruyko/wide-residual-networks). We use large batch sizes
in order to speed up training with many more GPU/TPUs. We also use l2 instead of
weight decay as weight decay is more difficult to implement in TensorFlow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import utils  # local file import
import six
from six.moves import range

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('base_learning_rate', 0.075,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('lr_warmup_epochs', 5,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', [60, 120, 160],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 1e-4, 'L2 regularization coefficient.')
flags.DEFINE_string('dataset', 'cifar10', 'Dataset: cifar10 or cifar100.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

Conv2D = functools.partial(  # pylint: disable=invalid-name
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal',
    kernel_regularizer=tf.keras.regularizers.l2(FLAGS.l2))


def basic_block(inputs, filters, strides):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.

  Returns:
    tf.Tensor.
  """
  x = inputs
  # Note using epsilon and momentum defaults from Torch which WRN paper used.
  y = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=strides)(y)
  y = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = Conv2D(filters, strides=1)(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters, kernel_size=1, strides=strides)(x)
  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks):
  """Group of residual blocks.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    num_blocks: Number of residual blocks.

  Returns:
    tf.Tensor.
  """
  x = basic_block(inputs, filters=filters, strides=strides)
  for _ in range(num_blocks - 1):
    x = basic_block(x, filters=filters, strides=1)
  return x


def wide_resnet(input_shape, depth, width_multiplier, num_classes, l2):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it uses the preactivation ResNet
  ordering (He et al., 2016) and accepts a width multiplier on the number of
  filters. Using three groups of residual blocks, the network maps spatial
  features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    depth: Total number of convolutional layers. "n" in WRN-n-k, as opposed to
      the maximum depth of the network counting non-conv layers like dense as
      with He et al. (2015).
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape)
  x = Conv2D(16, strides=1)(inputs)
  x = group(x, filters=16 * width_multiplier, strides=1, num_blocks=num_blocks)
  x = group(x, filters=32 * width_multiplier, strides=2, num_blocks=num_blocks)
  x = group(x, filters=64 * width_multiplier, strides=2, num_blocks=num_blocks)
  x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(
      num_classes,
      kernel_initializer='he_normal',
      kernel_regularizer=tf.keras.regularizers.l2(l2),
      bias_regularizer=tf.keras.regularizers.l2(l2))(x)
  return tf.keras.Model(inputs=inputs, outputs=x)


def main(argv):
  del argv  # unused arg
  tf.enable_v2_behavior()
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  def train_input_fn(ctx):
    """Sets up local (per-core) dataset batching."""
    dataset = utils.load_distributed_dataset(
        split=tfds.Split.TRAIN,
        name=FLAGS.dataset,
        batch_size=FLAGS.per_core_batch_size,
        drop_remainder=True,
        use_bfloat16=FLAGS.use_bfloat16)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset

  def test_input_fn(ctx):
    """Sets up local (per-core) dataset batching."""
    dataset = utils.load_distributed_dataset(
        split=tfds.Split.TEST,
        name=FLAGS.dataset,
        batch_size=FLAGS.per_core_batch_size,
        drop_remainder=True,
        use_bfloat16=FLAGS.use_bfloat16)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset

  train_dataset = strategy.experimental_distribute_datasets_from_function(
      train_input_fn)
  test_dataset = strategy.experimental_distribute_datasets_from_function(
      test_input_fn)
  ds_info = tfds.builder(FLAGS.dataset).info

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  with strategy.scope():
    logging.info('Building ResNet model')
    model = wide_resnet(input_shape=ds_info.features['image'].shape,
                        depth=28,
                        width_multiplier=10,
                        num_classes=ds_info.features['label'].num_classes,
                        l2=FLAGS.l2)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    # TODO(trandustin):
    lr_decay_epochs = [np.floor(FLAGS.train_epochs / 200 * start_epoch)
                       for start_epoch in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        # decay_epochs=FLAGS.lr_decay_epochs,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_nll = tf.keras.metrics.Mean('train_nll', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy', dtype=tf.float32)
    test_nll = tf.keras.metrics.Mean('test_nll', dtype=tf.float32)
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'test_accuracy', dtype=tf.float32)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries/'))

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      train_loss.update_state(loss)
      train_nll.update_state(negative_log_likelihood)
      train_accuracy.update_state(labels, logits)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, probs))
      test_nll.update_state(negative_log_likelihood)
      test_accuracy.update_state(labels, probs)

    strategy.experimental_run_v2(step_fn, args=(next(iterator),))

  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    with summary_writer.as_default():
      for step in range(steps_per_epoch):
        train_step(train_iterator)

        current_step = epoch * steps_per_epoch + step
        max_steps = steps_per_epoch * FLAGS.train_epochs
        time_elapsed = time.time() - start_time
        steps_per_sec = float(current_step) / time_elapsed
        eta_seconds = (max_steps - current_step) / (steps_per_sec + 1e-7)
        message = ('{:.1f}% completion, at step {:d}. {:.1f} steps/s. '
                   'ETA: {:.0f} min'.format(100 * current_step / max_steps,
                                            current_step,
                                            steps_per_sec,
                                            eta_seconds / 60))
        if step % 20 == 0:
          logging.info(message)

      tf.summary.scalar('train/loss',
                        train_loss.result(),
                        step=epoch + 1)
      tf.summary.scalar('train/negative_log_likelihood',
                        train_nll.result(),
                        step=epoch + 1)
      tf.summary.scalar('train/accuracy',
                        train_accuracy.result(),
                        step=epoch + 1)
      logging.info('Train Loss: %s, Accuracy: %s%%',
                   round(float(train_loss.result()), 4),
                   round(float(train_accuracy.result() * 100), 2))

      train_loss.reset_states()
      train_nll.reset_states()
      train_accuracy.reset_states()

      test_iterator = iter(test_dataset)
      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        test_step(test_iterator)
      tf.summary.scalar('test/negative_log_likelihood',
                        test_nll.result(),
                        step=epoch + 1)
      tf.summary.scalar('test/accuracy',
                        test_accuracy.result(),
                        step=epoch + 1)
      logging.info('Test NLL: %s, Accuracy: %s%%',
                   round(float(test_nll.result()), 4),
                   round(float(test_accuracy.result() * 100), 2))

      test_nll.reset_states()
      test_accuracy.reset_states()

    if (epoch + 1) % 20 == 0:
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

if __name__ == '__main__':
  app.run(main)
