import sys
import os.path
import glob
import random
import h5py
import numpy as np
import math
import string
#from matplotlib import pyplot
from functools import partial
from itertools import product
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Conv1D, MaxPool1D, Flatten
from keras.models import Model
from keras.constraints import maxnorm
from keras.losses import mean_squared_error
from sklearn.utils import class_weight
import keras
from keras import backend as K
import tensorflow as tf
import freetype
from sidebearings import safe_glyphs, loadfont, samples, get_m_width
from settings import augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, trust_zeros, mu, binfunction, kern_bins, files
from auxiliary import bigram_frequency, mse_penalizing_miss, create_class_weight, hinged_min_error

np.set_printoptions(precision=3, suppress=True)

# Design the network:
def drop(x): return Dropout(dropout_rate)(x)
def relu(x, layers=1, nodes=32):
  for _ in range(1,layers):
    x = Dense(nodes, activation='relu', kernel_initializer='uniform')(x)
  return x

print("Building network")

inputs = []
nets = []

for n in input_names:
  input_ = Input(shape=(samples,1), dtype='float32', name=n)
  inputs.append(input_)
  conv = Conv1D(2,2,activation='relu')(input_)
  pool = MaxPool1D(pool_size=2)(conv)
  flat = Flatten()(pool)
  # net = drop(input_)
  net = flat
  nets.append(net)

x = keras.layers.concatenate(nets)
# x = drop(relu(x, layers=depth,nodes=width))
x = drop(Dense(1024, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(512, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(256, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(128, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(64, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(128, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(256, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(512, activation='relu', kernel_initializer='uniform')(x))
x = drop(Dense(1024, activation='relu', kernel_initializer='uniform')(x))

if regress:
  kernvalue = Dense(1, activation="linear")(x)
else:
  kernvalue =  Dense(kern_bins, activation='softmax')(x)

if os.path.exists("kernmodel.hdf5"):
  model = keras.models.load_model("kernmodel.hdf5")
else:
  model = Model(inputs=inputs, outputs=[kernvalue])

  print("Compiling network")

  opt = keras.optimizers.adam(lr=init_lr)

  if regress:
    loss = 'mean_squared_error'
    metrics = []
  else:
    # loss = 'categorical_crossentropy'
    loss = mse_penalizing_miss
    metrics = ['accuracy']
  model.compile(loss=loss, metrics=metrics, optimizer=opt)


# Trains the NN given a font and its associated kern dump

checkpointer = keras.callbacks.ModelCheckpoint(filepath='kernmodel.hdf5', verbose=0, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_decay, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=2, min_lr=0)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

kern_input = []
sample_weights = []
input_tensors = {}
for n in input_names:
  input_tensors[n] = []

input_tensors["mwidth"] = []

def do_a_font(path, kerndump):
  loutlines, routlines, kernpairs = loadfont(path,kerndump)
  mwidth = get_m_width(path)
  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

  def add_entry(left, right):
    input_tensors["mwidth"].append(mwidth)

    if "minsumdist" in input_tensors:
      input_tensors["minsumdist"].append(np.min(rightcontour(left)+leftcontour(right)))

    if "nton" in input_tensors:
      input_tensors["nton"].append(np.min(rightcontour("n")+leftcontour("n")))

    if "otoo" in input_tensors:
      input_tensors["otoo"].append(np.min(rightcontour("o")+leftcontour("o")))

    if "leftofl" in input_tensors:
      input_tensors["leftofl"].append(leftcontour(left))
    if "rightofl" in input_tensors:
      input_tensors["rightofl"].append(rightcontour(left))

    if "leftofr" in input_tensors:
      input_tensors["leftofr"].append(leftcontour(right))
    if "rightofr" in input_tensors:
      input_tensors["rightofr"].append(rightcontour(right))

    if "leftofn" in input_tensors:
      input_tensors["leftofn"].append(leftcontour("n"))
    if "rightofn" in input_tensors:
      input_tensors["rightofn"].append(rightcontour("n"))
    if "leftofo" in input_tensors:
      input_tensors["leftofo"].append(leftcontour("o"))
    if "rightofo" in input_tensors:
      input_tensors["rightofo"].append(rightcontour("o"))
    if "leftofH" in input_tensors:
      input_tensors["leftofH"].append(leftcontour("H"))
    if "rightofH" in input_tensors:
      input_tensors["rightofH"].append(rightcontour("H"))
    if "leftofO" in input_tensors:
      input_tensors["leftofO"].append(leftcontour("O"))
    if "rightofO" in input_tensors:
      input_tensors["rightofO"].append(rightcontour("O"))

    if right in kernpairs[left]:
      kern = kernpairs[left][right]/mwidth
    else:
      kern = 0

    if regress:
      kern_input.append(kern)
    else:
      kern_input.append(binfunction(kern))

    sample_weights.append(bigram_frequency(left,right))

  for left in safe_glyphs:
    for right in safe_glyphs:
      if right in kernpairs[left] or trust_zeros:
        add_entry(left,right)

for i in files:
  print(i)
  do_a_font(i,i+".kerndump")

if not regress:
  kern_input = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
else:
  kern_input = np.array(kern_input)

for n in input_names:
  input_tensors[n] = np.array(input_tensors[n])

input_tensors["mwidth"] = np.array(input_tensors["mwidth"])
#Augment data
if augmentation > 0:
  for n in input_names:
    t = input_tensors[n]
    out = None
    augs = [t]
    for i in range(0,augmentation):
      aug = t + np.random.randint(-2, high=2, size=t.shape) / np.expand_dims(input_tensors["mwidth"],axis=2)
      augs.append(aug)
    input_tensors[n] = np.concatenate(augs)

  if regress:
    kern_input = np.tile(kern_input,1+augmentation)
  else:
    kern_input = np.tile(kern_input,(1+augmentation,1))

  sample_weights = np.tile(sample_weights,1+augmentation)

for n in input_names:
  input_tensors[n] = np.expand_dims(input_tensors[n], axis=2)

if regress:
  class_weight = None
else:
  # Correct for class frequency discrepancy
  # (Many more zero kerns than positive kerns)
  counts = kern_input.sum(axis=0)
  print(counts)
  class_weight = create_class_weight(dict(enumerate(kern_input.sum(axis=0))),mu)
  print(counts * list(class_weight.values()))

history = model.fit(input_tensors, kern_input,
  sample_weight = sample_weights,
  class_weight = class_weight,
  batch_size=batch_size, epochs=5000, verbose=1, callbacks=[
  earlystop,
  checkpointer,
  reduce_lr,
  tensorboard
],shuffle = True,
  validation_split=0.2, initial_epoch=0)



#pyplot.plot(history.history['val_loss'])
#pyplot.show()

