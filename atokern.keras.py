import sys
import os.path
import glob
import random
import numpy as np
import math
import string
import tensorflow as tf
from sidebearings import safe_glyphs, loadfont, samples
from settings import output_path, generate, mirroring, covnet, augmentation, batch_size, input_names, regress, threeway, trust_zeros, binfunction, kern_bins, training_files, validation_files, all_pairs, width, depth, max_per_font, is_bin
from model import model, callback_list
from auxiliary import bigram_frequency

import keras

safe_glyphs = list(safe_glyphs)

class_weights = [1] * kern_bins

def howmany(font_files, full=False):
  def fill_a_font(path):
    this_font = 0
    count = 0
    kerndump = path+".kerndump"
    loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
    for left in set(safe_glyphs):
      for right in set(safe_glyphs):
        if right in kernpairs[left]:
           o = kernpairs[left][right]/mwidth
        else:
           o = 0
        k =binfunction(o)
        this_font = this_font + 1
        class_weights[k]=class_weights[k]+1
        count = count+1
        if this_font > max_per_font:
          return this_font
    for left in set(safe_glyphs):
      for right in kernpairs[left]:
        if not right in safe_glyphs:
          if right in kernpairs[left]:
             o = kernpairs[left][right]/mwidth
          else:
             o = 0
          k =binfunction(o)
          this_font = this_font + 1
          class_weights[k]=class_weights[k]+1
          count = count+1
          if this_font > max_per_font:
            return this_font

    return count

  count = 0
  for path in font_files:
    count += fill_a_font(path)

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
  print(is_bin)
  if is_bin >= 0:
    kern_input = (np.array(kern_input) == is_bin).astype(int)
  elif not regress:
    kern_input = keras.utils.to_categorical(kern_input, num_classes=kern_bins)
  else:
    kern_input = np.array(kern_input)
  input_tensors["mwidth"] = np.array(input_tensors["mwidth"])
  for n in input_names:
    input_tensors[n] = np.array(input_tensors[n])
    if perturb:
      input_tensors[n] = input_tensors[n] + np.random.randint(-2, high=2, size=input_tensors[n].shape) / np.expand_dims(input_tensors["mwidth"],axis=2)
    if covnet:
      input_tensors[n] = np.expand_dims(input_tensors[n], axis=2)
  return kern_input, input_tensors

def add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, sample_weights, mirrored=False):
  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

  input_tensors["mwidth"].append(mwidth)
  sample_weights.append(bigram_frequency(left,right))

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
    kern_input.append(binfunction(kern)) # Ordinal regression
  else:
    kern_input.append(binfunction(kern))

def not_generator(font_files, perturb = False, full=False):
  random.shuffle(font_files)
  kern_input = []
  input_tensors = {}
  sample_weights = []
  for n in input_names:
    input_tensors[n] = []
  input_tensors["mwidth"] = []
  def add_a_font(path):
    this_font = 0
    random.shuffle(safe_glyphs)
    kerndump = path+".kerndump"
    loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
    for left in set(safe_glyphs):
      for right in set(safe_glyphs):
        if this_font > max_per_font:
          return
        if not left in routlines or not left in loutlines:
          print("Font %s claimed to have glyph %s but no outlines found" % (path,left))
          sys.exit(1)
        if not right in routlines or not right in loutlines:
          print("Font %s claimed to have glyph %s but no outlines found" % (path,right))
          sys.exit(1)
        add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, sample_weights)
        this_font = this_font + 1
        if mirroring: 
          this_font = this_font + 1
          add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, sample_weights, mirrored=True)
    for left in set(safe_glyphs):
      for right in kernpairs[left]:
        if not right in safe_glyphs:
          if this_font > max_per_font:
            return
          if not left in routlines or not left in loutlines:
            print("Font %s claimed to have glyph %s but no outlines found" % (path,left))
            sys.exit(1)
          if not right in routlines or not right in loutlines:
            print("Font %s claimed to have glyph %s but no outlines found" % (path,right))
            sys.exit(1)
          add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, sample_weights)
          this_font = this_font + 1
          if mirroring: 
            this_font = this_font + 1
            add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, sample_weights, mirrored=True)


  for path in font_files:
    print(path)
    add_a_font(path)
  print("Prepping entries")
  kern_input, input_tensors = prep_entries(kern_input, input_tensors, perturb)
  print("Done")
  return kern_input, input_tensors, sample_weights

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
        for left in kernpairs:
          for right in kernpairs[left]:
            if not right in safe_glyphs:
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

print("Shuffling")
print("Training")
if generate:
	history = model.fit_generator(generator(training_files, perturb = True, full = True),
	  steps_per_epoch = steps,
	  class_weight = class_weights,
	  epochs=6000, verbose=1, callbacks=callback_list,
	  validation_steps=val_steps,
	  validation_data=generator(validation_files, full=True))
else:
	kern_input, input_tensors, sample_weights = not_generator(training_files, perturb = False, full = True)
	# val_kern, val_tensors = not_generator(validation_files, perturb = True, full = True)

	history = model.fit(input_tensors, kern_input,
	   class_weight = class_weights,
     sample_weight = np.array(sample_weights),
	   batch_size=batch_size, epochs=6000, verbose=1, callbacks=callback_list,shuffle = True, 
           validation_split=0.2
           #validation_data=(val_tensors, val_kern)
           )

	#pyplot.plot(history.history['val_loss'])
	#pyplot.show()

model.save("output/kernmodel-final.hdf5")
