import sys
import os.path
import glob
import random
import h5py
import numpy as np
# from matplotlib import pyplot

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.constraints import maxnorm
import keras

import freetype
from sidebearings import safe_glyphs, loadfont, samples
epoch = 0

# Hyperparameters. These are all guesses. (Samples should be OK.)
zero_supression = 0.9

def drop(x): return Dropout(0.2)(x)
def relu(x, layers=1):
  for _ in range(1,layers):
    x = Dense(512, activation='relu')(x)
  return x

# Design the network:
print("Building network")
contour_input = Input(shape=(samples,), dtype='float32', name='contour')
contour_shape = relu(drop(contour_input), layers=5)

nn_input = Input(shape=(samples,), dtype='float32', name='nn')
nn_shape = relu(drop(nn_input), layers=5)

oo_input = Input(shape=(samples,), dtype='float32', name='oo')
oo_shape = relu(drop(oo_input), layers=5)

nR_input = Input(shape=(samples,), dtype='float32', name='nR')
nR_shape = relu(drop(oo_input), layers=5)

Ln_input = Input(shape=(samples,), dtype='float32', name='Ln')
Ln_shape = relu(drop(oo_input), layers=5)

x = keras.layers.concatenate([contour_shape, nn_shape, oo_shape,nR_shape, Ln_shape])
# x=contour_shape
x = drop(relu(x, layers=2))
x = Dense(512, activation="relu")(x)
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

model = Model(inputs=[contour_input, nn_input, oo_input, nR_input, Ln_input], outputs=[kernvalue])
# model = Model(inputs=[l_input, r_input], outputs=[kernvalue])
print("Compiling network")

opt = keras.optimizers.adam(lr=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# Trains the NN given a font and its associated kern dump
kern_input = []
nn = []
oo = []
leftRight = []
leftN = []
nRight = []

def do_a_font(path, kerndump, epoch):
  loutlines, routlines, kernpairs = loadfont(path,kerndump)
  face = freetype.Face(path)
  face.set_char_size( 64 * face.units_per_EM )
  n = face.get_name_index("n")
  face.load_glyph(n, freetype.FT_LOAD_RENDER |
                            freetype.FT_LOAD_TARGET_MONO)
  nwidth = face.glyph.metrics.horiAdvance / 64
  print("N width:", nwidth)
  def contour_between(left, right):
    return routlines[left] + routlines[right]

  for left in safe_glyphs:
    for right in safe_glyphs:
      if kernpairs[left][right] != 0 or random.random() > zero_supression:
        nn.append(contour_between('n','n') /nwidth)
        oo.append(contour_between('o','o') / nwidth)
        leftRight.append(contour_between(left, right) / nwidth)
        kern_input.append(bin_kern(kernpairs[left][right]) / nwidth)
        leftN.append(contour_between(left,'n') / nwidth)
        nRight.append(contour_between('n',right) / nwidth)

files = glob.glob("./kern-dump/*.?tf")
epochn = 0
for i in files:
  print(i)
  do_a_font(i,i+".kerndump", epochn)

kerncats = keras.utils.to_categorical(kern_input, num_classes=kern_bins)

checkpointer = keras.callbacks.ModelCheckpoint(filepath='kernmodel.hdf5', verbose=1, save_best_only=True)
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=20, verbose=1, mode='auto')

history = model.fit({
  "nn": np.array(nn),
  "oo": np.array(oo) ,
  "contour": np.array(leftRight) ,
  "nR": np.array(nRight),
  "Ln": np.array(leftN)
  }, kerncats,
  batch_size=32, epochs=2000, verbose=1, callbacks=[
  earlystop,
  checkpointer
],shuffle = True,
  validation_split=0.2, initial_epoch=0)

# pyplot.plot(history.history['acc'])
# pyplot.show()

