import sys
import os.path
import glob
import random
import h5py
import numpy as np
import math
import string
#from matplotlib import pyplot

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.constraints import maxnorm
import keras

import freetype
from sidebearings import safe_glyphs, loadfont, samples, get_m_width
epoch = 0

# Hyperparameters. These are all guesses.
augmentation = 2
batch_size = 256
depth = 5
width = 512
dropout_rate = 0.3
init_lr = 0.001

regress = False
threeway = True
files = glob.glob("kern-dump/*.?tf")

def drop(x): return Dropout(dropout_rate)(x)
def relu(x, layers=1, nodes=32):
  for _ in range(1,layers):
    x = Dense(nodes, activation='relu', kernel_initializer='uniform')(x)
  return x

# Design the network:
print("Building network")
input_names = [
"rightofl", "leftofr",
"rightofn", "leftofo"]
inputs = []
nets = []

for n in input_names:
  input_ = Input(shape=(samples,), dtype='float32', name=n)
  inputs.append(input_)
  net = drop(input_)
  nets.append(net)

x = keras.layers.concatenate(nets)
x = relu(x, layers=depth,nodes=width)

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
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
  model.compile(loss=loss, metrics=metrics, optimizer=opt)


# Trains the NN given a font and its associated kern dump

checkpointer = keras.callbacks.ModelCheckpoint(filepath='kernmodel.hdf5', verbose=0, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

kern_input = []
input_tensors = {}
upper = [i for i in string.ascii_uppercase]
lower = [i for i in string.ascii_lowercase]
for n in input_names:
  input_tensors[n] = []

def do_a_font(path, kerndump, epoch):
  loutlines, routlines, kernpairs = loadfont(path,kerndump)
  mwidth = get_m_width(path)
  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

  def add_entry(left, right,wiggle):
    if "minsumdist" in input_tensors:
      input_tensors["minsumdist"].append(np.min(rightcontour(left)+leftcontour(right)+2*wiggle/mwidth))

    if "nton" in input_tensors:
      input_tensors["nton"].append(np.min(rightcontour("n")+leftcontour("n")))

    if "otoo" in input_tensors:
      input_tensors["otoo"].append(np.min(rightcontour("o")+leftcontour("o")))

    if "leftofl" in input_tensors:
      input_tensors["leftofl"].append(leftcontour(left))
    if "rightofl" in input_tensors:
      input_tensors["rightofl"].append(rightcontour(left)+wiggle/mwidth)

    if "leftofr" in input_tensors:
      input_tensors["leftofr"].append(leftcontour(right)+wiggle/mwidth)
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

    if right in kernpairs[left]:
      kern = kernpairs[left][right]
    else:
      kern = 0
    kern = (kern-2*wiggle)/mwidth
    # kern = random.randint(-20,20)
    if regress:
      kern_input.append(kern)
    else:
      kern_input.append(binfunction(kern))

  for left in safe_glyphs:
    for right in safe_glyphs:
      if right in kernpairs[left]:
        for w in range(-augmentation,1+augmentation,1):
          add_entry(left,right,w)

epochn = 0
for i in files:
  print(i)
  do_a_font(i,i+".kerndump", epochn)

if not regress:
  kern_input2 = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
  bins = kern_input2.sum(axis=0)
  bins.fill(np.min(bins))
  selections = []
  for i in range(0,kern_input2.shape[0]-1):
    if bins[kern_input[i]] >= 0:
      selections.append(i)
    bins[kern_input[i]] = bins[kern_input[i]]-1

  kern_input = kern_input2[selections]
  for n in input_names:
    input_tensors[n] = np.array(input_tensors[n])[selections]
  print(kern_input.sum(axis=0))
else:
  for n in input_names:
    input_tensors[n] = np.array(input_tensors[n])

history = model.fit(input_tensors, kern_input,
  batch_size=batch_size, epochs=5000, verbose=1, callbacks=[
  earlystop,
  checkpointer,
  reduce_lr,
  tensorboard
],shuffle = True,
  validation_split=0.2, initial_epoch=0)



#pyplot.plot(history.history['val_loss'])
#pyplot.show()

