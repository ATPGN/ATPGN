"""
This tutorial shows how to generate adversarial examples using FGSM
and train a model using adversarial training with TensorFlow.
The original paper can be found at:
https://arxiv.org/abs/1412.6572
"""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
import os
import json
import math

from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.compat import flags
from cleverhans.dataset import CIFAR10 
from dataset import CIFAR100
from cleverhans.loss import CrossEntropy
from cleverhans.train import train
from cleverhans.utils_tf import model_eval
from cleverhans.serial import save, load
from models.PGN import PGN, PGNLoss
from train import train_with_PGN
from evaluate import evaluate

FLAGS = flags.FLAGS

def train(train_start=0, train_end=60000, test_start=0, test_end=10000):
  """
  :param train_start: index of first training set example
  :param train_end: index of last training set example
  :param test_start: index of first test set example
  :param test_end: index of last test set example
  """

  assert FLAGS.train_type in ['PGN', 'Naive', 'Goodfellow', 'Madry']
  if FLAGS.train_type == 'Naive':
    save_dir = 'save/%s/%s/Naive'%(FLAGS.dataset,FLAGS.train_type)
  elif FLAGS.train_type == 'PGN':
    save_dir = 'save/%s/%s/cl_%.2f'%(FLAGS.dataset,FLAGS.train_type,FLAGS.l2_constraint)
  else:
    save_dir = 'save/%s/%s/eps_%.2f'%(FLAGS.dataset,FLAGS.train_type,FLAGS.eps)

  if not os.path.exists(save_dir):
      os.makedirs(save_dir)

  # Set TF random seed to improve reproducibility
  tf.set_random_seed(1234)
  np.random.seed(1234)

  # Create TF session
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  # Get CIFAR10 data
  if FLAGS.dataset == 'CIFAR100':
    data = CIFAR100(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  elif FLAGS.dataset == 'CIFAR10':
    data = CIFAR10(train_start=train_start, train_end=train_end,
                 test_start=test_start, test_end=test_end)
  else:
    raise ValueError('FLAGS.dataset should be "CIFAR100" or "CIFAR10"')


  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(FLAGS.batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_test, y_test = data.get_set('test')
  s = np.arange(x_test.shape[0])
  x_test = x_test[s]
  y_test = y_test[s]

  # Use Image Parameters
  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # Define input TF placeholder
  x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                        nchannels))
  y = tf.placeholder(tf.float32, shape=(None, nb_classes))

  # Train an MNIST model
  train_params = {
    'nb_epochs': FLAGS.nb_epochs,
    'batch_size': FLAGS.batch_size,
    'learning_rate': FLAGS.learning_rate,
  }
  eval_params = {'batch_size': FLAGS.batch_size}
  goodfellow_params = {
    'eps': FLAGS.eps,
    'ord': 2,
    'clip_min': 0.,
    'clip_max': 1.
  }
  madry_params = {
    'eps': FLAGS.eps,
    'eps_iter': FLAGS.eps/FLAGS.nb_iter*5/3,
    'nb_iter': FLAGS.nb_iter,
    'ord': 2,
    'clip_min': 0.,
    'clip_max': 1.
  }

  rng = np.random.RandomState([2019, 9, 9])

  def do_eval(preds, x_set, y_set, report_text):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    print('Test accuracy on %s examples: %0.4f' % (report_text, acc))
    return acc

  model = PGN('AllconvNet', nb_classes=nb_classes, nb_filters=FLAGS.nb_filters, input_shape=[img_rows, img_cols, nchannels], batch_size=FLAGS.batch_size, input_min=0.0, input_max=1.0)
  preds = model.get_logits(x)
  goodfellow_l2 = FastGradientMethod(model, sess=sess)
  madry_l2 = ProjectedGradientDescent(model, sess=sess)

  if FLAGS.train_type == 'PGN':
    loss = PGNLoss(model, L2_constraint=FLAGS.l2_constraint, adv_coeff=FLAGS.adv_coeff)
  elif FLAGS.train_type == 'Naive':
    loss = CrossEntropy(model)
  elif FLAGS.train_type == 'Goodfellow':
    loss = CrossEntropy(model, attack=goodfellow_l2, adv_coeff=FLAGS.adv_coeff, pass_y=True, attack_params=goodfellow_params)
  elif FLAGS.train_type == 'Madry':
    loss = CrossEntropy(model, attack=madry_l2, adv_coeff=FLAGS.adv_coeff, pass_y=True, attack_params=madry_params)

  def evaluate(epoch):
    acc = do_eval(preds, x_test, y_test, 'clean')
    return acc
  
  train_with_PGN(sess, model, loss, train_type=FLAGS.train_type,
        dataset_train=dataset_train, dataset_size=dataset_size,
        evaluate=evaluate, args=train_params, rng=rng, use_ema=True,
        classifier_var_list=model.get_classifier_params(),
        generator_var_list=model.get_generator_params(), save_dir=save_dir)


def main(argv=None):
  if FLAGS.evaluate:
    evaluate(FLAGS.dataset)
  else:
    train()


if __name__ == '__main__':
  flags.DEFINE_boolean('save', True,
                       'save model or not')
  flags.DEFINE_boolean('evaluate', False,
                       'evaluate or not')
  flags.DEFINE_integer('nb_filters', 64,
                       'Model size multiplier')
  flags.DEFINE_integer('nb_epochs', 200,
                       'Number of epochs to train model')
  flags.DEFINE_integer('batch_size', 128,
                       'Size of training batches')
  flags.DEFINE_integer('nb_iter', 10,
                       'Size of training batches')
  flags.DEFINE_float('learning_rate', 5e-4,
                     'Learning rate for training')
  flags.DEFINE_float('eps', 0.02, 'eps for fgm params')
  flags.DEFINE_float('l2_constraint', 70,
                     'l2 constraint for training')
  flags.DEFINE_float('adv_coeff', 1.0,
                     'adv_coefficient for training')
  flags.DEFINE_string('train_type', 'PGN',
                     'adv_coefficient for training')
  flags.DEFINE_string('dataset', 'CIFAR100',
                     'dataset for training')

  tf.app.run()
