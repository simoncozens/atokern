from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K
from functools import partial
from itertools import product
import numpy as np
import math

from settings import binfunction, kern_bins, false_negative_penalty, false_positive_penalty

import csv
bigrams = {}
with open('bigrams.csv','r') as tsvin:
  tsvin = csv.reader(tsvin, delimiter='\t',quoting=csv.QUOTE_NONE)
  for row in tsvin:
    l,r,freq = row[0], row[1], row[2]
    if not l in bigrams:
      bigrams[l]={}
    bigrams[l][r] = freq

def bigram_frequency(l,r):
  if l in glyphname_to_ascii:
    l = glyphname_to_ascii[l]
  if r in glyphname_to_ascii:
    r = glyphname_to_ascii[r]
  try:
    w = float(bigrams[l][r])/82801640.0
  except Exception as e:
    return 1
  return float(w)/82801640.0

glyphname_to_ascii = {
   "one":"1", "two":"2", "three":"3", "four":"4", "five":"5", "six":"6",
   "seven":"7", "eight":"8", "nine":"9", "zero":"0",
   "period":".", "comma":",", "colon":":"
}

def create_class_weight(labels_dict,mu=0.15):
  total = sum(labels_dict.values())
  keys = labels_dict.keys()
  class_weight = dict()
  for key in keys:
      score = math.log(mu*total/float(labels_dict[key]))
      class_weight[key] = score if score > 1.0 else 1.0
  return class_weight

def hinged_min_error(y_true, y_pred):
  mse = mean_squared_error(y_true, y_pred)
  return tf.cast(K.less(y_pred, 0.),tf.float32) * mse * mse + mse

def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask
w_array = np.ones((kern_bins,kern_bins))
w_array[:,binfunction(0)]=false_negative_penalty
w_array[binfunction(0),:]=false_positive_penalty
w_array[binfunction(0),binfunction(0)]=1
mse_penalizing_miss = partial(w_categorical_crossentropy, weights=w_array)
mse_penalizing_miss.__name__ ='mse_penalizing_miss'