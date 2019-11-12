"""
Multi-replica synchronous training


NOTE: This module is much more free to change than many other modules
in CleverHans. CleverHans is very conservative about changes to any
code that affects the output of benchmark tests (attacks, evaluation
methods, etc.). This module provides *model training* functionality
not *benchmarks* and thus is free to change rapidly to provide better
speed, accuracy, etc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import time
import warnings

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans import canary
from cleverhans.utils import _ArgsWrapper, create_logger
from cleverhans.utils import safe_zip
from cleverhans.utils_tf import infer_devices
from cleverhans.utils_tf import initialize_uninitialized_global_variables
from cleverhans.serial import save

_logger = create_logger("train")
_logger.setLevel(logging.INFO)


def train_with_PGN(sess, model, loss, train_type='naive', evaluate=None, args=None,
          rng=None, classifier_var_list=None, generator_var_list=None, save_dir=None,
          fprop_args=None, optimizer=None, use_ema=False, ema_decay=.998,
          loss_threshold=1e10, dataset_train=None, dataset_size=None):
  """
  Run (optionally multi-replica, synchronous) training to minimize `loss`
  :param sess: TF session to use when training the graph
  :param loss: tensor, the loss to minimize
  :param evaluate: function that is run after each training iteration
                   (typically to display the test/validation accuracy).
  :param args: dict or argparse `Namespace` object.
               Should contain `nb_epochs`, `learning_rate`,
               `batch_size`
  :param rng: Instance of numpy.random.RandomState
  :param var_list: Optional list of parameters to train.
  :param fprop_args: dict, extra arguments to pass to fprop (loss and model).
  :param optimizer: Optimizer to be used for training
  :param use_ema: bool
      If true, uses an exponential moving average of the model parameters
  :param ema_decay: float or callable
      The decay parameter for EMA, if EMA is used
      If a callable rather than a float, this is a callable that takes
      the epoch and batch as arguments and returns the ema_decay for
      the current batch.
  :param loss_threshold: float
      Raise an exception if the loss exceeds this value.
      This is intended to rapidly detect numerical problems.
      Sometimes the loss may legitimately be higher than this value. In
      such cases, raise the value. If needed it can be np.inf.
  :param dataset_train: tf Dataset instance.
      Used as a replacement for x_train, y_train for faster performance.
    :param dataset_size: integer, the size of the dataset_train.
  :return: True if model trained
  """

  # Check whether the hardware is working correctly
  canary.run_canary()
  args = _ArgsWrapper(args or {})
  fprop_args = fprop_args or {}

  # Check that necessary arguments were given (see doc above)
  # Be sure to support 0 epochs for debugging purposes
  if args.nb_epochs is None:
    raise ValueError("`args` must specify number of epochs")
  if optimizer is None:
    if args.learning_rate is None:
      raise ValueError("Learning rate was not given in args dict")
  assert args.batch_size, "Batch size was not given in args dict"
  assert dataset_train and dataset_size, "dataset_train or dataset_size was not given"

  if rng is None:
    rng = np.random.RandomState()

  if optimizer is None:
    optimizer = tf.train.AdamOptimizer(learning_rate = args.learning_rate)
  else:
    if not isinstance(optimizer, tf.train.Optimizer):
      raise ValueError("optimizer object must be from a child class of "
                       "tf.train.Optimizer")

  grads_classifier = []
  if train_type == 'PGN':
    grads_generator = []
  xs = []
  ys = []
  data_iterator = dataset_train.make_one_shot_iterator().get_next()
  x_train, y_train = sess.run(data_iterator)

  devices = infer_devices()
  for device in devices:
    with tf.device(device):
      x = tf.placeholder(x_train.dtype, (None,) + x_train.shape[1:])
      y = tf.placeholder(y_train.dtype, (None,) + y_train.shape[1:])
      xs.append(x)
      ys.append(y)
      if train_type == 'PGN':
        loss_classifier, loss_generator = loss.fprop(x, y, **fprop_args)
      else:
        loss_classifier = loss.fprop(x, y, **fprop_args)
      grads_classifier.append(optimizer.compute_gradients(loss_classifier, var_list=classifier_var_list))
      if train_type == 'PGN':
        grads_generator.append(optimizer.compute_gradients(loss_generator, var_list=generator_var_list))

  num_devices = len(devices)
  print("num_devices: ", num_devices)

  grad_classifier = avg_grads(grads_classifier)
  if train_type == 'PGN':
    grad_generator = avg_grads(grads_generator)
  # Trigger update operations within the default graph (such as batch_norm).
  with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_step = optimizer.apply_gradients(grad_classifier)
    if train_type == 'PGN':
      with tf.control_dependencies([train_step]):
        train_step = optimizer.apply_gradients(grad_generator)

  var_list = classifier_var_list
  if train_type == 'PGN':
    var_list += generator_var_list
  if use_ema:
    ema = tf.train.ExponentialMovingAverage(decay=ema_decay)
    with tf.control_dependencies([train_step]):
      train_step = ema.apply(var_list)
    # Get pointers to the EMA's running average variables
    avg_params = [ema.average(param) for param in var_list]
    # Make temporary buffers used for swapping the live and running average
    # parameters
    tmp_params = [tf.Variable(param, trainable=False)
                  for param in var_list]
    # Define the swapping operation
    param_to_tmp = [tf.assign(tmp, param)
                    for tmp, param in safe_zip(tmp_params, var_list)]
    with tf.control_dependencies(param_to_tmp):
      avg_to_param = [tf.assign(param, avg)
                      for param, avg in safe_zip(var_list, avg_params)]
    with tf.control_dependencies(avg_to_param):
      tmp_to_avg = [tf.assign(avg, tmp)
                    for avg, tmp in safe_zip(avg_params, tmp_params)]
    swap = tmp_to_avg

  batch_size = args.batch_size

  assert batch_size % num_devices == 0
  device_batch_size = batch_size // num_devices

  sess.run(tf.global_variables_initializer())
  best_acc = 0.0

  for epoch in xrange(args.nb_epochs):
    nb_batches = int(math.ceil(float(dataset_size) / batch_size))
    prev = time.time()
    for batch in range(nb_batches):
      x_train_shuffled, y_train_shuffled = sess.run(data_iterator)
      start, end = 0, batch_size
      feed_dict = dict()
      for dev_idx in xrange(num_devices):
        cur_start = start + dev_idx * device_batch_size
        cur_end = start + (dev_idx + 1) * device_batch_size
        feed_dict[xs[dev_idx]] = x_train_shuffled[cur_start:cur_end]
        feed_dict[ys[dev_idx]] = y_train_shuffled[cur_start:cur_end]

      
      _, loss_classifier_numpy = sess.run([train_step, loss_classifier], feed_dict=feed_dict)

      if np.abs(loss_classifier_numpy) > loss_threshold:
        raise ValueError("Extreme loss_classifier during training: ", loss_classifier_numpy)
      if np.isnan(loss_classifier_numpy) or np.isinf(loss_classifier_numpy):
        raise ValueError("NaN/Inf loss_classifier during training")
    cur = time.time()
    _logger.info("Epoch " + str(epoch) + " took " +
                 str(cur - prev) + " seconds")
    if evaluate is not None:
      if use_ema:
        sess.run(swap)
      r_value = evaluate(epoch)

      if use_ema:
        sess.run(swap)
  if use_ema:
    sess.run(swap)

  with sess.as_default():
    save_path = os.path.join(save_dir,'model.joblib')
    save(save_path, model)

  return True


def avg_grads(tower_grads):
  """Calculate the average gradient for each shared variable across all
  towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been
     averaged across all towers.

  Modified from this tutorial: https://tinyurl.com/n3jr2vm
  """
  if len(tower_grads) == 1:
    return tower_grads[0]
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = [g for g, _ in grad_and_vars]

    # Average over the 'tower' dimension.
    grad = tf.add_n(grads) / len(grads)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    assert all(v is grad_and_var[1] for grad_and_var in grad_and_vars)
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads
