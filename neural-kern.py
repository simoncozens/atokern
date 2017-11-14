import freetype
import numpy as np
import pickle
import sys
import os.path
import glob
import random
from math import copysign
from sidebearings import safe_glyphs, loadfont, samples, get_m_width
from settings import augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, hinged_min_error, mse_penalizing_miss
import keras

np.set_printoptions(precision=3, suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#safe_glyphs = ["A", "V", "W", "T", "Y", "H", "O", "n", "i", "h", "e"]

# Design the network:
print("Loading network")
model = keras.models.load_model("kernmodel.hdf5", custom_objects={'hinged_min_error': hinged_min_error, 'mse_penalizing_miss': mse_penalizing_miss})

def bin_to_label3(value, mwidth):
  if value == 0: return "-"
  if value == 1: return "0"
  if value == 2: return "+"

def bin_to_label(value, mwidth):
  rw = 800
  scale = mwidth/rw
  if value == 0:
    low = "-inf"; high = int(-150 * scale)
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
    return "0"
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
    low = int(50 * scale); high = int(inf * scale)
  return str(low)+" - "+str(high)

if threeway:
  binfunction = bin_to_label3
else:
  binfunction = bin_to_label

input_tensors = {}
for n in input_names:
  input_tensors[n] = []

# Trains the NN given a font and its associated kern dump
def do_a_font(path):
  mwidth = get_m_width(path)
  print("Loading font")
  loutlines, routlines, _ = loadfont(path, None)

  def leftcontour(letter):
    return np.array(loutlines[letter])/mwidth
  def rightcontour(letter):
    return np.array(routlines[letter])/mwidth

  for left in sorted(safe_glyphs):
    for right in sorted(safe_glyphs):
      wiggle = 0
      if "minsumdist" in input_tensors:
        input_tensors["minsumdist"].append(np.min(rightcontour(left)+leftcontour(right)+2*wiggle/mwidth))
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
      if "leftofH" in input_tensors:
        input_tensors["leftofH"].append(leftcontour("H"))
      if "rightofH" in input_tensors:
        input_tensors["rightofH"].append(rightcontour("H"))
      if "leftofO" in input_tensors:
        input_tensors["leftofO"].append(leftcontour("O"))
      if "rightofO" in input_tensors:
        input_tensors["rightofO"].append(rightcontour("O"))

  for n in input_names:
    input_tensors[n] = np.array(input_tensors[n])
    input_tensors[n] = np.expand_dims(input_tensors[n], axis=2)

  predictions = model.predict(input_tensors)
  loop = 0
  if not regress:
    classes = np.argmax(predictions, axis=1)

  for left in sorted(safe_glyphs):
    for right in sorted(safe_glyphs):
      if regress:
        prediction = int(predictions[loop] * mwidth)
        print(left, right, prediction)
      else:
        prediction = classes[loop]
        if binfunction(prediction, mwidth) != "0":
          print(left, right, binfunction(prediction, mwidth), "p=", int(100*predictions[loop][classes[loop]]), "%")

      loop = loop + 1

do_a_font(sys.argv[1])
