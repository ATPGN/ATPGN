import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math

from dataset import CIFAR100
from cleverhans.dataset import CIFAR10
from cleverhans.serial import load
from cleverhans.compat import flags
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.utils_tf import model_eval
from cleverhans.utils import set_log_level

def evaluate(dataset='CIFAR100'):
  batch_size = 128
  test_num=10000
  defense_list = ['Naive', 'Goodfellow', 'Madry', 'PGN']
  model_path_list=[]
  for defense in defense_list:
    for i in os.listdir('save/%s/%s'%(dataset,defense)):
      if os.path.exists('save/%s/%s/%s/model.joblib'%(dataset,defense,i)):
        model_path_list.append('save/%s/%s/%s/model.joblib'%(dataset,defense,i))
  
  if dataset == 'CIFAR100':
    data = CIFAR100(test_start=0, test_end=test_num)
    x_test, y_test = data.get_set('test')
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 100))
  elif dataset == 'CIFAR10':
    data = CIFAR10(test_start=0, test_end=test_num)
    x_test, y_test = data.get_set('test')
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
  
  sess = tf.Session()
  
  cw_params = {
    'batch_size': 128,
    'clip_min': 0.,
    'clip_max': 1.,
    'max_iterations':100,
    'y':y
  }
  
  eval_params = {'batch_size': batch_size}
  
  def do_eval(preds, x_set, y_set, report_text):
    acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
    print('Test accuracy on %s: %0.4f' %(report_text,acc))
    return acc
  
  def get_adv_x_numpy(adv_x, attack_success_index, x_set, y_set):
    result = []
    result_index = []
    nb_batches = int(math.ceil(float(len(x_set)) / batch_size))
    X_cur = np.zeros((batch_size,)+x_set.shape[1:], dtype=x_set.dtype)
    Y_cur = np.zeros((batch_size,)+y_set.shape[1:], dtype=y_set.dtype)
    for batch in range(nb_batches):
      start = batch * batch_size
      end = min(len(x_set), start+batch_size)
      cur_batch_size = end-start
      X_cur[:cur_batch_size] = x_set[start:end]
      Y_cur[:cur_batch_size] = y_set[start:end]
      feed_dict = {x:X_cur, y:Y_cur}
      adv_x_numpy, success_index = sess.run([adv_x, attack_success_index], feed_dict = feed_dict)
      result.append(adv_x_numpy[:cur_batch_size])
      result_index.append(success_index[:cur_batch_size])
    return np.concatenate(result, axis=0), np.concatenate(result_index, axis=0)
  
  print(model_path_list)
  acc_dict={}
  l2mean_dict={}
  for model_path in model_path_list:
    defense = model_path.split('/')[2]
    if not defense in acc_dict:
      acc_dict[defense]=[]
    if not defense in l2mean_dict:
      l2mean_dict[defense]=[]
  
    if os.path.exists(os.path.join(os.path.dirname(model_path),'cash_result')):
      with open(os.path.join(os.path.dirname(model_path),'cash_result'), 'r') as f:
        cash_result_str = f.read()
        acc, l2mean, model_create_time = cash_result_str.split(",")
  
      if int(model_create_time) == int(os.path.getctime(model_path)):
        acc_dict[defense].append(float(acc))
        l2mean_dict[defense].append(float(l2mean))
        print(model_path, acc, l2mean)
        continue
  
    with sess.as_default():
      model = load(model_path)
      
    attack_model = CarliniWagnerL2(model, sess=sess)
    attack_params = cw_params
    
    preds = model.get_logits(x)
    acc = do_eval(preds, x_test[:test_num], y_test[:test_num], 'DEFENSE : %s'%defense)
    adv_x = attack_model.generate(x, **attack_params)
    preds_adv = model.get_logits(adv_x)
    attack_success_index = tf.math.not_equal(tf.argmax(preds_adv, axis=-1),tf.argmax(y, axis=-1))
    adv_x_numpy, success_index = get_adv_x_numpy(adv_x, attack_success_index, x_test[:test_num], y_test[:test_num])
    print('C&W attack success_rate = %f'%np.mean(success_index))


    l2mean = np.mean(np.sqrt(np.sum(np.power(adv_x_numpy[success_index]-x_test[:test_num][success_index],2), axis=(1,2,3))))
  
    acc_dict[defense].append(acc)
    l2mean_dict[defense].append(l2mean)
    print(model_path, acc, l2mean)
    with open(os.path.join(os.path.dirname(model_path),'cash_result'), 'w') as f:
      f.write('%.4f,%.4f,%d'%(acc, l2mean, os.path.getctime(model_path)))
  
  for defense in defense_list:
    if not defense in l2mean_dict:
        continue

    l2mean_dict[defense] = np.array(l2mean_dict[defense])
    acc_dict[defense] = np.array(acc_dict[defense])
    arg_l2mean_dict = np.argsort(l2mean_dict[defense])
    l2mean_dict[defense] = l2mean_dict[defense][arg_l2mean_dict]
    acc_dict[defense] = acc_dict[defense][arg_l2mean_dict]
    plt.plot(l2mean_dict[defense], acc_dict[defense], '-o', label=defense)
  plt.legend()
  plt.xlabel('$\\rho_{cw}$')
  plt.ylabel('benign accuracy')
  plt.title("RESULT FOR %s"%dataset)

  fig_save_dir = 'evaluate/%s'%dataset
  if not os.path.exists(fig_save_dir):
    os.makedirs(fig_save_dir)
  plt.savefig('%s/robustness-curve.png'%fig_save_dir)
  
