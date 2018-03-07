import sys
import os.path
import glob
import random
import h5py
import numpy as np
import math
import string
import keras
import pickle

from sidebearings import safe_glyphs, loadfont, samples
from settings import output_path, generate, mirroring, covnet, augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, trust_zeros, mu, binfunction, kern_bins, training_files, validation_files, all_pairs, width, depth, max_per_font
from auxiliary import bigram_frequency, mse_penalizing_miss, create_class_weight, hinged_min_error

np.set_printoptions(precision=3, suppress=False)

safe_glyphs = list(safe_glyphs)

class_weights = [1] * kern_bins

import sqlite3

sqlite_file = 'batches.sqlite'
conn = sqlite3.connect(sqlite_file)
c = conn.cursor()

c.execute('CREATE TABLE training (rowid PRIMARY KEY, path, left,right, k , mirror)')
c.execute('CREATE TABLE validation (rowid PRIMARY KEY, path, left,right, k , mirror)')

row = 0

def fill(font_files,table):
  def ins(row,path,left,right,k,mirror):
    s = 'INSERT INTO {tn} (rowid, path,left,right,k,mirror) VALUES ({rowid}, "{path}", "{left}", "{right}", {k}, {mirror})'.\
        format(tn=table, rowid=row, path=path,left=left,right=right,k=k, mirror=mirror)
    c.execute(s)


  def fill_a_font(path):
    global row
    count = 0
    kerndump = path+".kerndump"
    loutlines, routlines, kernpairs, mwidth = loadfont(path,kerndump)
    for left in (set(kernpairs)|set(safe_glyphs)):
      for right in (set(kernpairs[left])|set(safe_glyphs)):
        if left in kernpairs and right in kernpairs[left]:
           o = kernpairs[left][right]/mwidth
        else:
           o = 0
        k =binfunction(o)
        count = count + 1
        if count > max_per_font:
          return
        ins(row,path, left, right, k, 0)
        row = row + 1
        if mirroring:
          ins(row,path, left, right, k, 1)
          row = row + 1
          if count > max_per_font:
            return
        class_weights[k]=class_weights[k]+1

  fileno = 0
  for path in font_files:
    fileno = fileno+1
    print("%s (%i/%i)" % (path, fileno, len(font_files)))
    fill_a_font(path)
  conn.commit()

fill(training_files, "training")
row = 0
fill(validation_files, "validation")


#conn.close()
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
print(class_weights)
print(np.sum(class_weights))
print("Baseline: ", class_weights[binfunction(0)] / np.sum(class_weights)*100, "%")
class_weights = np.sum(class_weights) / np.array(class_weights)
pickle.dump(class_weights, open("class_weights.pickle","wb"))
print("Shuffling...")

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
  if not (left in routlines) or not (right in loutlines):
    return

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


print("Building batches...")

def build_batch(foo, batch, steps, batch_type):
  print("Batch %s %i/%i" % (batch_type, foo, math.ceil(steps)))
  cache = {}
  kern_input = []
  input_tensors = {}
  for n in input_names:
    input_tensors[n] = []
  input_tensors["mwidth"] = []
  for element in batch:
    path, left, right, kernvalue, mirror = element
    kerndump = path+".kerndump"
    if not path in cache:
      cache[path] = loadfont(path,kerndump)
    loutlines, routlines, kernpairs, mwidth = cache[path]
    if mirror:
      add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input, mirrored=True)
    else:
      add_entry(left,right, mwidth, kernpairs, loutlines, routlines, input_tensors, kern_input)
  kern_input, input_tensors = prep_entries(kern_input, input_tensors, False)
  pickle.dump({"kern_input":kern_input, "input_tensors": input_tensors}, open(batch_type+"-batches/%08i.pickle" % foo,"wb"))

foo = 0
indices = np.array(range(0,train_count))
np.random.shuffle(indices)
f = np.array_split(indices,math.ceil(steps))

for index_batch in f:
  c.execute('SELECT path, left, right, k, mirror FROM training WHERE rowid in ('+(",".join(map(str,index_batch)))+')')
  batch = c.fetchall()
  print(index_batch)
  foo = foo + 1
  build_batch(foo, batch, math.ceil(steps), "training")

foo = 0
indices = np.array(range(0,val_count))
np.random.shuffle(indices)
f = np.array_split(indices,math.ceil(val_steps))
for index_batch in f:
  c.execute('SELECT path, left, right, k, mirror FROM validation WHERE rowid in ('+(",".join(map(str,index_batch)))+')')
  batch = c.fetchall()
  foo = foo + 1
  build_batch(foo, batch, math.ceil(val_steps), "validation")

