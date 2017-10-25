import sys
import os.path
import glob
import random
import h5py
import numpy as np
import math
import string
# from matplotlib import pyplot

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.constraints import maxnorm
import keras

import freetype
from sidebearings import safe_glyphs, loadfont, samples, get_m_width
epoch = 0
np.set_printoptions(precision=3, suppress=True)

# Hyperparameters. These are all guesses.
zero_supression = 0.3
repeat_supression = 0.2

def drop(x): return Dropout(0.1)(x)
def relu(x, layers=1, nodes=32):
  for _ in range(1,layers):
    x = Dense(nodes, activation='relu')(x)
  return x

regress = False
threeway = True

# Design the network:
print("Building network")
input_names = ["rightofl", "leftofr"]
inputs = []
nets = []

for n in input_names:
  input_ = Input(shape=(samples,), dtype='float32', name=n)
  inputs.append(input_)
  nets.append(relu(input_,layers=3,nodes=128))

x = keras.layers.concatenate(nets)

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

model = Model(inputs=inputs, outputs=[kernvalue])

print("Compiling network")

opt = keras.optimizers.adam()
if regress:
  loss = 'mean_squared_error'
else:
  loss = 'categorical_crossentropy'
model.compile(loss=loss, metrics=['accuracy'],
              optimizer=opt)

# Trains the NN given a font and its associated kern dump

checkpointer = keras.callbacks.ModelCheckpoint(filepath='kernmodel.hdf5', verbose=1, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
kern_input = []
input_tensors = {}
upper = [i for i in string.ascii_uppercase]
lower = [i for i in string.ascii_lowercase]

def do_a_font(path, kerndump, epoch):
  loutlines, routlines, kernpairs = loadfont(path,kerndump)
  mwidth = get_m_width(path)
  for n in input_names:
    input_tensors[n] = []
  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

  def add_entry(left, right,wiggle):
    if "leftofl" in input_tensors:
      input_tensors["leftofl"].append(leftcontour(left)/mwidth)
    if "rightofl" in input_tensors:
      input_tensors["rightofl"].append(rightcontour(left)+wiggle/mwidth)

    if "leftofr" in input_tensors:
      input_tensors["leftofr"].append(leftcontour(right)+wiggle/mwidth)
    if "rightofr" in input_tensors:
      input_tensors["rightofr"].append(rightcontour(right)/mwidth)

    kern = (kernpairs[left][right]-2*wiggle)/mwidth
    if regress:
      kern_input.append(kern)
    else:
      kern_input.append(binfunction(kern))

  for left in upper:
    for right in upper+lower:
      if kernpairs[left][right] != 0 or random.random() < zero_supression:
        for w in range(-2,2):
          add_entry(left,right,w)

  for left in lower:
    for right in lower:
      if kernpairs[left][right] != 0 or random.random() < zero_supression:
        for w in range(-2,2):
          add_entry(left,right,w)

files = glob.glob("./kern-dump/Sou*egular.?tf")
epochn = 0
for i in files:
  print(i)
  do_a_font(i,i+".kerndump", epochn)

if not regress:
  kern_input = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
  print(kern_input.sum(axis=0))
  print(kern_input.sum(axis=0).sum(axis=0))

for n in input_names:
  input_tensors[n] = np.array(input_tensors[n])

history = model.fit(input_tensors, kern_input,
  batch_size=1, epochs=200, verbose=1, callbacks=[
  earlystop
  # checkpointer
],shuffle = True,
  validation_split=0.2, initial_epoch=0)



# pyplot.plot(history.history['acc'])
# pyplot.show()

