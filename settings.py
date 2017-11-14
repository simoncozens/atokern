from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K
from functools import partial
from itertools import product
import numpy as np
import glob

files = glob.glob("kern-dump/*.?tf")

# Hyperparameters. These are all guesses.
augmentation = 3
batch_size = 1024
# depth = 10
# width = 8
dropout_rate = 0.3
init_lr = 0.001
lr_decay = 0.5
mu = 0.3
# We predicted 0 but it wasn't
false_negative_penalty = 10
# It was 0 but we said it wasn't
false_positive_penalty = 5

input_names = [
"rightofl", "leftofr",
"rightofn", "leftofo",
"rightofH", "leftofO"
]
regress = False
threeway = False
trust_zeros = True

def bin_kern3(value):
  if value < -5/800: return 0
  if value > 5/800: return 2
  return 1

def bin_kern(value):
  rw = 800
  if value < -150/rw: return 0
  if value < -100/rw: return 1
  if value < -70/rw: return 2
  if value < -50/rw: return 3
  if value < -45/rw: return 4
  if value < -40/rw: return 5
  if value < -35/rw: return 6
  if value < -30/rw: return 7
  if value < -25/rw: return 8
  if value < -20/rw: return 9
  if value < -15/rw: return 10
  if value < -10/rw: return 11
  if value < -5/rw: return 12
  if value < 0: return 13
  if value == 0: return 14
  if value > 50/rw: return 25
  if value > 45/rw: return 24
  if value > 40/rw: return 23
  if value > 35/rw: return 22
  if value > 30/rw: return 21
  if value > 25/rw: return 20
  if value > 20/rw: return 19
  if value > 15/rw: return 18
  if value > 10/rw: return 17
  if value > 5/rw: return 16
  if value > 0: return 15

if threeway:
  kern_bins = 3
  binfunction = bin_kern3
else:
  kern_bins = 26
  binfunction = bin_kern

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
