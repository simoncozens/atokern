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
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from keras.models import Model
from keras.constraints import maxnorm
#from keras.losses import mean_squared_error
from sklearn.utils import class_weight
import keras
from keras import backend as K
import tensorflow as tf
# import freetype
from sidebearings import safe_glyphs, loadfont, samples
from settings import generate, mirroring, covnet, augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, trust_zeros, mu, binfunction, kern_bins, training_files, validation_files, all_pairs, width, depth
from auxiliary import bigram_frequency, mse_penalizing_miss, create_class_weight, hinged_min_error

np.set_printoptions(precision=3, suppress=False)

# Design the network:
def drop(x): return Dropout(dropout_rate)(x)
def relu(x, layers=1, nodes=32):
  for _ in range(1,layers):
    x = Dense(nodes, activation='relu', kernel_initializer='he_normal')(x)
  return x

print("Building network")

inputs = []
nets = []

for n in input_names:
  if covnet:
    input_ = Input(shape=(samples,1), dtype='float32', name=n)
  else:
    input_ = Input(shape=(samples,), dtype='float32', name=n)
  inputs.append(input_)
  if covnet:
    conv = Conv1D(16,2,activation='relu')(input_)
    flat = Flatten()(conv)
    net = flat
  else:
    net = input_
  nets.append(net)

x = keras.layers.concatenate(nets)
x = drop(x)
x = relu(x, layers=depth,nodes=width)
x = drop(x)
#x = Dense(1024, activation='relu', kernel_initializer='uniform')(x)
#x = Dense(512, activation='relu', kernel_initializer='uniform')(x)
#x = Dense(256, activation='relu', kernel_initializer='uniform')(x)
#x = Dense(128, activation='relu', kernel_initializer='uniform')(x)
#x = Dense(64, activation='relu', kernel_initializer='uniform')(x)
#x = drop(Dense(128, activation='relu', kernel_initializer='uniform')(x))
#x = drop(Dense(256, activation='relu', kernel_initializer='uniform')(x))
#x = drop(Dense(512, activation='relu', kernel_initializer='uniform')(x))
#x = drop(Dense(1024, activation='relu', kernel_initializer='uniform')(x))

if regress:
  kernvalue = Dense(1, activation="linear")(x)
else:
  kernvalue =  Dense(kern_bins, activation='softmax')(x)

if os.path.exists("kernmodel.hdf5"):
  model = keras.models.load_model("kernmodel.hdf5", custom_objects={'hinged_min_error': hinged_min_error, 'mse_penalizing_miss': mse_penalizing_miss})
else:
  model = Model(inputs=inputs, outputs=[kernvalue])

  print("Compiling network")

  opt = keras.optimizers.adam(lr=init_lr)

  if regress:
    loss = 'mean_squared_error'
    metrics = []
  else:
    #loss = 'categorical_crossentropy'
    loss = mse_penalizing_miss
    metrics = ['accuracy']
  model.compile(loss=loss, metrics=metrics, optimizer=opt)


# Trains the NN given a font and its associated kern dump

checkpointer = keras.callbacks.ModelCheckpoint(filepath='kernmodel.hdf5', verbose=0, save_best_only=True, monitor="val_loss")
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=50, verbose=1, mode='auto')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_decay, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=1, min_lr=0)
tensorboard = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

safe_glyphs = list(safe_glyphs)

class_weights = [1] * kern_bins

def howmany(font_files, full=False):
  count = 0
  for path in font_files:
    kerndump = path+".kerndump"
    loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
    for left in safe_glyphs:
      for right in safe_glyphs:
        if right in kernpairs[left]:
          k =binfunction(kernpairs[left][right]/mwidth)
        else:
          k=binfunction(0)
        class_weights[k]=class_weights[k]+1
    count = count + (len(safe_glyphs) * len(safe_glyphs))
    if all_pairs and full:
      for left in kernpairs:
        for right in (set(kernpairs[left])|set(safe_glyphs)):
          if right in kernpairs[left]:
             o = kernpairs[left][right]/mwidth
          else:
             o = 0
          k =binfunction(o)
          class_weights[k]=class_weights[k]+1
          count = count+1
  return count

print("Counting...")
steps = math.ceil(howmany(training_files, full=all_pairs) / batch_size) * augmentation
val_steps = math.ceil(howmany(validation_files) / batch_size)
if mirroring:
   steps = steps * 2
   val_steps = val_steps * 2
print(steps," steps")
print(class_weights)
print(np.sum(class_weights))

print("Baseline: ", class_weights[binfunction(0)] / np.sum(class_weights)*100, "%")

class_weights = np.sum(class_weights) / np.array(class_weights)

def prep_entries(kern_input, input_tensors, perturb):
  if not regress:
    kern_input = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
  else:
    kern_input = np.array(kern_input)
  input_tensors["mwidth"] = np.array(input_tensors["mwidth"])
  for n in input_names:
    input_tensors[n] = np.array(input_tensors[n])
    # if perturb:
      # input_tensors[n] = input_tensors[n] + np.random.randint(-2, high=2, size=input_tensors[n].shape) / np.expand_dims(input_tensors["mwidth"],axis=2)
    if covnet:
      input_tensors[n] = np.expand_dims(input_tensors[n], axis=2)
  return kern_input, input_tensors

def add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=False):
  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

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
    if mirrored:
      input_tensors["rightofl"].append(leftcontour(right))
    else:
      input_tensors["rightofl"].append(rightcontour(left))

  if "leftofr" in input_tensors:
    if mirrored:
      input_tensors["leftofr"].append(rightcontour(left))
    else:
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

def not_generator(font_files, perturb = False, full=False):
  random.shuffle(font_files)
  kern_input = []
  input_tensors = {}
  for n in input_names:
    input_tensors[n] = []
  input_tensors["mwidth"] = []

  for path in font_files:
    print(path)
    random.shuffle(safe_glyphs)
    kerndump = path+".kerndump"
    loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)

    for left in safe_glyphs:
      for right in safe_glyphs:
        if right in kernpairs[left] or trust_zeros:
          add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
          if mirroring: add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
    if full and all_pairs:
      # Also add entries for *all* defined kern pairs
      for left in kernpairs:
        for right in (set(kernpairs[left])|set(safe_glyphs)):
          add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
          if mirroring: add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
  kern_input, input_tensors = prep_entries(kern_input, input_tensors, perturb)
  return kern_input, input_tensors

def generator(font_files, perturb = False, full=False):
  while True:
    random.shuffle(font_files)
    step = 0
    for path in font_files:
      random.shuffle(safe_glyphs)
      kerndump = path+".kerndump"
      loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
      kern_input = []
      input_tensors = {}
      for n in input_names:
        input_tensors[n] = []
      input_tensors["mwidth"] = []

      for left in safe_glyphs:
        for right in safe_glyphs:
          if right in kernpairs[left] or trust_zeros:
            add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
            if mirroring: add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
            if len(kern_input) >= batch_size:
               kern_input, input_tensors = prep_entries(kern_input, input_tensors, perturb)
               step = step + 1
               yield(input_tensors, kern_input)
               kern_input = []
               entries = []
               input_tensors = {}
               for n in input_names:
                 input_tensors[n] = []
               input_tensors["mwidth"] = []

      if full and all_pairs:
        # Also add entries for *all* defined kern pairs
        for left in kernpairs:
          for right in (set(kernpairs[left])|set(safe_glyphs)):
            add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
            if mirroring: add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
            if len(kern_input) >= batch_size:
               kern_input, input_tensors = prep_entries(kern_input, input_tensors, perturb)
               yield(input_tensors, kern_input)
               kern_input = []
               input_tensors = {}
               for n in input_names:
                 input_tensors[n] = []
               input_tensors["mwidth"] = []
    kern_input, input_tensors = prep_entries(kern_input, input_tensors, perturb)
    if len(kern_input) > 0:
       yield(input_tensors, kern_input)

# if regress:
# class_weight = None
# else:

print("Training")
if generate:
	history = model.fit_generator(generator(training_files, perturb = True, full = True),
	  steps_per_epoch = steps,
	  class_weight = class_weights,
	  epochs=5000, verbose=1, callbacks=[
	  earlystop,
	  checkpointer,
	  #reduce_lr,
	  tensorboard
	],shuffle = True,
	  validation_steps=val_steps,
	  validation_data=generator(validation_files), initial_epoch=0)
else:

	kern_input, input_tensors = not_generator(training_files, perturb = True, full = True)
	val_kern, val_tensors = not_generator(validation_files, perturb = True, full = False)

	history = model.fit(input_tensors, kern_input,
	   class_weight = class_weights,
	   batch_size=batch_size, epochs=5000, verbose=1, callbacks=[
	    earlystop,
	    checkpointer,
	    reduce_lr,
	    tensorboard
	    ],shuffle = True, validation_data=(val_tensors, val_kern))

	#pyplot.plot(history.history['val_loss'])
	#pyplot.show()

