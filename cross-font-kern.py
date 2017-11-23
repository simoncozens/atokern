import freetype
import numpy as np
import pickle
import sys
import os.path
import glob
import random
from math import copysign
from sidebearings import get_m_width, get_one_glyph
from settings import augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway
from auxiliary import mse_penalizing_miss
import keras

np.set_printoptions(precision=3, suppress=True)
(path1, glyph1, path2, glyph2) = sys.argv[1::]

# Design the network:
model = keras.models.load_model("kernmodel.hdf5", custom_objects={'mse_penalizing_miss': mse_penalizing_miss})

def bin_to_label3(value, mwidth):
  if value == 0: return "-"
  if value == 1: return "0"
  if value == 2: return "+"

def bin_to_label(value, mwidth):
  rw = 800
  scale = mwidth/rw
  if value == 0:
    low = int(-300 * scale); high = int(-150 * scale)
  if value == 1:
    low = int(-150 * scale); high = int(-100 * scale)
  if value == 2:
    low = int(-100 * scale); high = int(-70 * scale)
  if value == 3:
    low = int(-70 * scale); high = int(-50 * scale)
  if value == 4:
    low = int(-50 * scale); high = int(-45 * scale)
  if value == 5:
    low = int(-45 * scale); high = int(-40 * scale)
  if value == 6:
    low = int(-40 * scale); high = int(-35 * scale)
  if value == 7:
    low = int(-35 * scale); high = int(-30 * scale)
  if value ==8:
    low = int(-30 * scale); high = int(-25 * scale)
  if value ==9:
    low = int(-25 * scale); high = int(-20 * scale)
  if value ==10:
    low = int(-20 * scale); high = int(-15 * scale)
  if value == 11:
    low = int(-15 * scale); high = int(-10 * scale)
  if value == 12:
    low = int(-10 * scale); high = int(-5 * scale)
  if value == 13:
    low = int(-5 * scale); high = int(-0 * scale)
  if value == 14:
    return 0
  if value == 15:
    low = int(0 * scale); high = int(5 * scale)
  if value == 16:
    low = int(5 * scale); high = int(10 * scale)
  if value == 17:
    low = int(10 * scale); high = int(15 * scale)
  if value == 18:
    low = int(15 * scale); high = int(20 * scale)
  if value == 19:
    low = int(20 * scale); high = int(25 * scale)
  if value == 20:
    low = int(25 * scale); high = int(30 * scale)
  if value == 21:
    low = int(30 * scale); high = int(35 * scale)
  if value == 22:
    low = int(35 * scale); high = int(40 * scale)
  if value == 23:
    low = int(40 * scale); high = int(45 * scale)
  if value == 24:
    low = int(45 * scale); high = int(50 * scale)
  if value == 25:
    low = int(50 * scale); high = int(100 * scale)
  return int(5 * round(float((low+high)/2)/5))

if threeway:
  binfunction = bin_to_label3
else:
  binfunction = bin_to_label

input_tensors = {}
for n in input_names:
  input_tensors[n] = []

mwidth1 = get_m_width(path1)
mwidth2 = get_m_width(path2)
_,rightofl = get_one_glyph(path1,glyph1)
_,rightofn = get_one_glyph(path1,"n")
_,rightofH = get_one_glyph(path1,"H")

leftofr,_ = get_one_glyph(path2,glyph2)
leftofo,_ = get_one_glyph(path2,"O")
leftofO,_ = get_one_glyph(path2,"O")

input_tensors["rightofl"].append(rightofl / mwidth1)
input_tensors["rightofn"].append(rightofn / mwidth1)
input_tensors["rightofH"].append(rightofH / mwidth1)

input_tensors["leftofr"].append(leftofr / mwidth2)
input_tensors["leftofo"].append(leftofo / mwidth2)
input_tensors["leftofO"].append(leftofO / mwidth2)

for n in input_names:
  input_tensors[n] = np.array(input_tensors[n])
  input_tensors[n] = np.expand_dims(input_tensors[n], axis=2)

predictions = model.predict(input_tensors)
loop = 0
classes = np.argmax(predictions, axis=1)

prediction = classes[0]
print(glyph1, glyph2, binfunction(prediction, (mwidth1+mwidth2)/2), "p=", int(100*predictions[0][classes[0]]), "%")
# print("\t", predictions[0])
