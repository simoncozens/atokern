from keras.losses import mean_squared_error
import tensorflow as tf
from keras import backend as K

# Hyperparameters. These are all guesses.
augmentation = 30
batch_size = 1024
# depth = 10
# width = 8
dropout_rate = 0.3
init_lr = 0.0001
lr_decay = 0.5
input_names = [
"rightofl", "leftofr",
"rightofn", "leftofo",
"rightofH", "leftofO"
]
regress = False
threeway = False
trust_zeros = False

def hinged_min_error(y_true, y_pred):
  mse = mean_squared_error(y_true, y_pred)
  return tf.cast(K.less(y_pred, 0.),tf.float32) * mse * mse + mse
