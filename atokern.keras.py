import sys
import os.path
import glob
import random
import h5py
import numpy as np
from matplotlib import pyplot

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.constraints import maxnorm
import keras

from sidebearings import safe_glyphs, loadfont
epoch = 0

# Hyperparameters. These are all guesses. (Samples should be OK.)
zero_supression = 0.9
samples = 100
n_l_inputs = samples
n_r_inputs = n_l_inputs

def drop(x): return Dropout(0.2)(x)
def relu(x): return Dense(32, activation='relu')(x)

# Design the network:
print("Building network")
l_input = Input(shape=(n_l_inputs,), dtype='float32', name='left')
lshape = Dropout(0.2)(l_input)
lshape = Dense(32, activation='relu')(lshape)

r_input = Input(shape=(n_r_inputs,), dtype='float32', name='right')
rshape = Dropout(0.2)(r_input)
lshape = Dense(32, activation='relu')(rshape)

n_left_input = Input(shape=(samples,), dtype='float32', name='left_n')
n_right_input = Input(shape=(samples,), dtype='float32', name='right_n')
nlshape = n_left_input
nrshape = n_right_input
nlshape = Dense(32,activation='relu')(nlshape)
nrshape = Dense(32,activation='relu')(nrshape)

o_left_input = Input(shape=(samples,), dtype='float32', name='left_o')
o_right_input = Input(shape=(samples,), dtype='float32', name='right_o')
olshape = o_left_input
orshape = o_right_input
olshape = Dense(32,activation='relu')(olshape)
orshape = Dense(32,activation='relu')(orshape)

x = keras.layers.concatenate([lshape, rshape])
x = Dense(8, activation="relu")(x)
kern_bins = 9

kernvalue =  Dense(kern_bins, activation='softmax')(x)

def bin_kern(value):
  if value < -50: return 0
  if value < -20: return 1
  if value < -10: return 2
  if value < 0: return 3
  if value == 0: return 4
  if value > 0: return 5
  if value > 10: return 6
  if value > 20: return 7
  if value > 50: return 8

model = Model(inputs=[l_input, r_input, n_left_input, n_right_input, o_left_input, o_right_input], outputs=[kernvalue])
# model = Model(inputs=[l_input, r_input], outputs=[kernvalue])
print("Compiling network")

opt = keras.optimizers.adam()
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# Trains the NN given a font and its associated kern dump
left_input = []
right_input = []
kern_input = []
o_left_input = []
o_right_input = []
n_left_input = []
n_right_input = []


def do_a_font(path, kerndump, epoch):
  loutlines, routlines, kernpairs = loadfont(path,kerndump)
  for left in safe_glyphs:
    for right in safe_glyphs:
      if kernpairs[left][right] != 0 or random.random() > zero_supression:
        left_input.append(routlines[left])
        right_input.append(loutlines[right])
        o_left_input.append(loutlines["o"])
        o_right_input.append(routlines["o"])
        n_left_input.append(loutlines["n"])
        n_right_input.append(routlines["n"])
        kern_input.append(bin_kern(kernpairs[left][right]))

files = glob.glob("./kern-dump/*.?tf")
epochn = 0
for i in files:
  print(i)
  do_a_font(i,i+".kerndump", epochn)

kerncats = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
history = model.fit({
  "left":  np.array(left_input),
  "right": np.array(right_input),
  "left_n": np.array(n_left_input),
  "right_n": np.array(n_right_input),
  "left_o": np.array(o_left_input),
  "right_o": np.array(o_right_input)
  }, kerncats,
  batch_size=32, epochs=50, verbose=1, callbacks=None,shuffle = True,
  validation_split=0.2, initial_epoch=0)

pyplot.plot(history.history['acc'])
pyplot.show()

model.save("kernmodel.hdf5")