import math
import tensorflow as tf

from cleverhans import initializers
from cleverhans.serial import NoRefModel
from cleverhans.loss import Loss

class PGN(NoRefModel):
  def __init__(self, scope, nb_classes, nb_filters, input_shape, batch_size, input_min, input_max, **kwargs):
    del kwargs
    super().__init__(scope, nb_classes, locals())
    self.nb_filters = nb_filters
    self.input_shape = input_shape
    self.batch_size = batch_size
    self.input_min = input_min
    self.input_max = input_max
    self.make_params()

  def get_classifier_params(self):
    out = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/classifier/")
    return out

  def get_generator_params(self):
    out = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope + "/generator/")
    return out

  def fprop(self, x, y=None, **kwargs):
    del kwargs
    out = {}
    conv_args = dict(activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'), kernel_size=3, padding='same')
    #conv_args = dict(activation=tf.nn.leaky_relu, kernel_initializer=initializers.HeReLuNormalInitializer, kernel_size=3, padding='same')
    convT_args = dict(kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'), kernel_size=3, strides=2, padding='same')
    dense_args = dict(activation=tf.nn.leaky_relu, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'))
    x_ = x

    with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE):
      log_resolution = int(round(math.log(self.input_shape[0]) / math.log(2)))
      with tf.variable_scope('classifier', reuse = tf.AUTO_REUSE):
        for scale in range(log_resolution -2):
          x_ = tf.layers.conv2d(x_, self.nb_filters << scale, **conv_args)
          x_ = tf.layers.conv2d(x_, self.nb_filters << (scale+1), **conv_args)
          x_ = tf.layers.average_pooling2d(x_, 2, 2)
        x_ = tf.layers.conv2d(x_, self.nb_classes, **conv_args)
        logits = tf.reduce_mean(x_, [1, 2])
        probs = tf.nn.softmax(logits=logits)
        out[self.O_LOGITS]=logits
        out[self.O_PROBS]=probs

      if y is not None:
        with tf.variable_scope('generator', reuse = tf.AUTO_REUSE):
          label_probs = tf.boolean_mask(out[self.O_PROBS], tf.cast(y, tf.bool))
          label_probs_mean = tf.reduce_mean(label_probs)
          grad_x, = tf.gradients(label_probs_mean, x)
          #grad_x, = tf.gradients(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits, axis=-1), x)
          #grad_x = grad_x*2*x*(1-x)

          #z = tf.random_uniform([tf.shape(x)[0], 100], -1, 1, name='z_train')

          l = tf.layers.dense(y, 1024, **dense_args)
          l = tf.layers.dense(tf.concat([l, y], 1), 64 * 2 * self.input_shape[0]//4 * self.input_shape[0]//4, **dense_args)
          l = tf.reshape(l, [-1, self.input_shape[0]//4, self.input_shape[0]//4, 64 * 2])

          y_ = y
          y_ = tf.reshape(y_, [-1, 1, 1, self.nb_classes])
          l = tf.concat([l, tf.tile(y_, [1, self.input_shape[0]//4, self.input_shape[0]//4, 1])], 3)
          l = tf.layers.conv2d_transpose(l, 64 * 2, activation=tf.nn.leaky_relu, **convT_args)
          
          l = tf.concat([l, tf.tile(y_, [1,self.input_shape[0]//2,self.input_shape[0]//2,1])], 3)
          l = tf.layers.conv2d_transpose(l, 64 * 2, activation=tf.nn.leaky_relu, **convT_args)

          l = tf.concat([l, grad_x, (x-self.input_min)/(self.input_max-self.input_min)*2-1.0], 3)
          l = tf.layers.conv2d(l, 64 * 2, **conv_args)
          l = tf.layers.conv2d(l, self.input_shape[2], activation=tf.identity, kernel_initializer=tf.keras.initializers.VarianceScaling(2.0, distribution='untruncated_normal'), kernel_size=3, padding='same')
          out['adv_image'] = (tf.math.tanh(tf.math.atanh(((x-self.input_min)/(self.input_max-self.input_min)*2-1.0)*0.999999)+l)+1.0)/2*(self.input_max-self.input_min)+self.input_min
          out['perturbation'] = out['adv_image']-x
          out['l2_loss'] = tf.reduce_sum(tf.pow(out['perturbation'],2), axis=[1,2,3])
          out['label_probs_mean'] = label_probs_mean

      return out

  def make_params(self):
    self.fprop(tf.placeholder(tf.float32, [32] + self.input_shape), y=tf.placeholder(tf.float32, [32,self.nb_classes]))

class PGNLoss(Loss):

  def __init__(self, model, adv_coeff=1.0, L2_constraint=1.0):
    super().__init__(model)
    self.adv_coeff=adv_coeff
    self.L2_constraint=L2_constraint

  def fprop(self, x, y):
    logits = self.model.get_logits(x)
    adv_img = self.model.get_layer(x,'adv_image',y=y)
    l2_loss = self.model.get_layer(x,'l2_loss',y=y)
    l2_distance = tf.math.sqrt(l2_loss)
    logits_adv = self.model.get_logits(adv_img)
    y = tf.stop_gradient(y)
    loss_clean = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits, axis=-1))#+self.model.get_layer(x, 'label_probs_mean', y=y)
    loss_adv = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = y, logits = logits_adv, axis=-1))#+self.model.get_layer(adv_img, 'label_probs_mean', y=y)
    loss_classifier = (1-self.adv_coeff)*loss_clean+self.adv_coeff*loss_adv
    loss_generator1 = -loss_adv#self.model.get_layer(adv_img, 'label_probs_mean', y=y)
    loss_generator2 = tf.reduce_mean(l2_loss)
    loss_generator = loss_generator1+self.L2_constraint*loss_generator2

    return loss_classifier, loss_generator
