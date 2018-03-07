import sys
import os.path
import numpy as np
import math
import keras
import tensorflow as tf
from sidebearings import safe_glyphs, loadfont, samples
from settings import output_path, generate, mirroring, covnet, augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, trust_zeros, mu, binfunction, kern_bins, training_files, validation_files, all_pairs, width, depth
import pickle
from keras.utils import Sequence

from model import model, callback_list
safe_glyphs = list(safe_glyphs)

import sqlite3

sqlite_file = 'batches.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

# c.execute('CREATE TABLE training (rowid PRIMARY KEY, path, left,right, k , mirror)')
# c.execute('CREATE TABLE validation (rowid PRIMARY KEY, path, left,right, k , mirror)')


# def fill(font_files,table):
#   def ins(row,path,left,right,k,mirror):
#     s = 'INSERT INTO {tn} (rowid, path,left,right,k,mirror) VALUES ({rowid}, "{path}", "{left}", "{right}", {k}, {mirror})'.\
#         format(tn=table, rowid=row, path=path,left=left,right=right,k=k, mirror=mirror)
#     c.execute(s)
#   row = 0
#   for path in font_files:
#     print(path)
#     intermediate_array = []
#     kerndump = path+".kerndump"
#     loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
#     if not all_pairs:
#       for left in safe_glyphs:
#         for right in safe_glyphs:
#           if right in kernpairs[left]:
#             k =binfunction(kernpairs[left][right]/mwidth)
#           else:
#             k=binfunction(0)
#           ins(row,path, left, right, k, 0)
#           row = row + 1
#           if mirroring:
#             ins(row,path, left, right, k, 1)
#             row = row + 1
#           class_weights[k]=class_weights[k]+1
#     else:
#       for left in (set(kernpairs)|set(safe_glyphs)):
#         for right in (set(kernpairs[left])|set(safe_glyphs)):
#           if left in kernpairs and right in kernpairs[left]:
#              o = kernpairs[left][right]/mwidth
#           else:
#              o = 0
#           k =binfunction(o)
#           ins(row,path, left, right, k, 0)
#           row = row + 1
#           if mirroring:
#             ins(row,path, left, right, k, 1)
#             row = row + 1
#           class_weights[k]=class_weights[k]+1
#   conn.commit()

# fill(training_files, "training")
# fill(validation_files, "validation")


# conn.close()
c.execute('SELECT count(*) FROM training')
train_count = c.fetchall()
train_count = train_count[0][0]
steps = train_count / batch_size

c.execute('SELECT count(*) FROM validation')
val_count = c.fetchall()
val_count = val_count[0][0]
val_steps = val_count / batch_size
print(steps," steps")
print(val_steps," validation steps")
# print(class_weights)
# print(np.sum(class_weights))
# print("Baseline: ", class_weights[binfunction(0)] / np.sum(class_weights)*100, "%")
# class_weights = np.sum(class_weights) / np.array(class_weights)
# pickle.dump(class_weights, open("class_weights.pickle","wb"))
# print("Shuffling...")

class_weights = pickle.load(open("class_weights.pickle","rb"))
# Build batches

def prep_entries(kern_input, input_tensors, perturb):
  if not regress:
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


from functools import lru_cache

@lru_cache(maxsize=64)
def getfont(path):
    kerndump = path+".kerndump"
    return loadfont(path,kerndump)

def build_batch(foo, batch, steps, batch_type):
  kern_input = []
  input_tensors = {}
  for n in input_names:
    input_tensors[n] = []
  input_tensors["mwidth"] = []
  for element in batch:
    path, left, right, kernvalue, mirror = element
    loutlines, routlines, kernpairs, mwidth = getfont(path)
    if mirror:
      add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
    else:
      add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
  kern_input, input_tensors = prep_entries(kern_input, input_tensors, False)
  return kern_input, input_tensors

class MySequence(Sequence):
  def __init__(self, cat,steps):
    self.cat = cat
    self.steps = steps

  def __len__(self):
    return math.ceil(self.steps)

  def __getitem__(self, idx):
    #ids = self.f[idx]
    #self.conn = sqlite3.connect(sqlite_file)
    #self.c = self.conn.cursor()
    #self.c.execute('SELECT path, left, right, k, mirror FROM training WHERE rowid in ('+(",".join(map(str,ids)))+')')
    #batch = self.c.fetchall()
    #kern_input, input_tensors = build_batch(idx, batch, math.ceil(steps), "training")
    obj  = pickle.load(open(self.cat+"-batches/%08i.pickle" % (idx+1),"rb"))
    return obj["input_tensors"], obj["kern_input"]

print("Training")
history = model.fit_generator(MySequence("training",steps),
  steps_per_epoch = steps,
  class_weight = class_weights,
  shuffle=False,
  use_multiprocessing=True,
  epochs=6000, verbose=1, callbacks=callback_list,
  validation_steps=val_steps,
  validation_data=MySequence("validation",val_steps), initial_epoch=2)
