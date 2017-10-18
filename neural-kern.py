import freetype
import numpy as np
import pickle
import sys
import os.path
import glob
import random
from math import copysign

from keras.layers import Input, Embedding, LSTM, Dense, Dropout
from keras.models import Model
from keras.constraints import maxnorm
import keras

np.set_printoptions(precision=3, suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

epoch = 0

# Hyperparameters. These are all guesses. (Samples should be OK.)

samples = 100
n_l_inputs = samples
n_r_inputs = n_l_inputs

# Design the network:
print("Loading network")
model = keras.models.load_model("kernmodel.hdf5")

def bin2label(value):
  # if value == 0: return "-"
  # if value == 1: return "0"
  # if value == 2: return "+"
  if value == 0: return "<-50"
  if value == 1: return "-25"
  if value == 2: return "-15"
  if value == 3: return "-5"
  if value == 4: return "0"
  if value == 5: return "5"
  if value == 6: return "15"
  if value == 7: return "25"
  if value == 8: return ">50"

safe_glyphs = set([
  "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
  "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
  "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
  "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
   "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", 
   "period", "comma", "colon"])

def unpack_mono_bitmap(bitmap):
  data = bytearray(bitmap.rows * bitmap.width)
  buff = bitmap._get_buffer()
  for y in range(bitmap.rows):
    for byte_index in range(bitmap.pitch):
      byte_value = buff[y * bitmap.pitch + byte_index]
      num_bits_done = byte_index * 8
      rowstart = y * bitmap.width + byte_index * 8
      for bit_index in range(min(8, bitmap.width - num_bits_done)):
        bit = byte_value & (1 << (7 - bit_index))
        data[rowstart + bit_index] = 1 if bit else 0
  return data

def bbox(outline):
  start, end = 0, 0
  VERTS = []
  # Iterate over each contour
  for i in range(len(outline.contours)):
      end    = outline.contours[i]
      points = outline.points[start:end+1]
      points.append(points[0])
      tags   = outline.tags[start:end+1]
      tags.append(tags[0])
      segments = [ [points[0],], ]
      for j in range(1, len(points) ):
          segments[-1].append(points[j])
          if tags[j] & (1 << 0) and j < (len(points)-1):
              segments.append( [points[j],] )
      verts = [points[0], ]
      for segment in segments:
          if len(segment) == 2:
              verts.extend(segment[1:])
          elif len(segment) == 3:
              verts.extend(segment[1:])
          else:
              verts.append(segment[1])
              for i in range(1,len(segment)-2):
                  A,B = segment[i], segment[i+1]
                  C = ((A[0]+B[0])/2.0, (A[1]+B[1])/2.0)
                  verts.extend([ C, B ])
              verts.append(segment[-1])
      VERTS.extend(verts)
      start = end+1
  VERTS = np.array(VERTS)
  x,y = VERTS[:,0], VERTS[:,1]
  VERTS[:,0], VERTS[:,1] = x, y

  xmin, xmax = x.min() /64, x.max() /64
  ymin, ymax = y.min() /64, y.max()/64
  return (xmin, xmax, ymin,ymax)

# Turn a glyph into a tensor of boundary samples
def glyph_to_sb(face, data, which="L"):
  glyph = face.glyph
  sb = []
  w, h = glyph.bitmap.width, glyph.bitmap.rows
  ascender = face.ascender
  lsb = glyph.metrics.horiBearingX / 64.0
  rsb = glyph.metrics.horiAdvance / 64.0 - (w + glyph.metrics.horiBearingX / 64.0)
  # print("Width: ", w)
  # print("Height: ", h)
  # print("LSB: ", lsb)
  # print("RSB: ", rsb)
  # print("Ascender", ascender)
  # print("Bearing Y", glyph.metrics.horiBearingY / 64.0)
  # print("Bbox", bbox(glyph.outline))
  (xmin, xmax, ymin,ymax) = bbox(glyph.outline)
  if which == "L":
    iterx = range(w)
    last = w-1
    const = rsb
  else:
    iterx = range(w-1,-1,-1)
    last = 0
    const = lsb

  for _ in range(ascender-int(glyph.metrics.horiBearingY / 64.0)):
    sb.append(int(const+w))

  for y in range(-int(ymin),h):
    for x in iterx:
      y2 = int(ymin)+y
      if data[w*y2+x] == 1 or x==last:
        if which == "L":
          sb.append(int(const+x))
        else:
          sb.append(int(const+(w-x)))
        break

  newsb = []
  i = 0
  for i in range(samples):
    sliceval = int(i*len(sb) / samples)
    newsb.append(sb[sliceval] / w)
  return newsb

# Trains the NN given a font and its associated kern dump
def do_a_font(path):
  face = freetype.Face(path)
  face.set_char_size( 64*500)
  loutlines = dict()
  routlines = dict()

  left_input = []
  right_input = []
  kern_input = []
  o_left_input = []
  o_right_input = []
  n_left_input = []
  n_right_input = []

  for g in safe_glyphs:
    glyphindex = face.get_name_index(g.encode("utf8"))
    if glyphindex:
      face.load_glyph(glyphindex, freetype.FT_LOAD_RENDER |
                                freetype.FT_LOAD_TARGET_MONO)
      data = unpack_mono_bitmap(face.glyph.bitmap)
      loutlines[g] = np.array(glyph_to_sb(face, data, which="L"))
      routlines[g] = np.array(glyph_to_sb(face, data, which="R"))

  for left in sorted(safe_glyphs):
    for right in sorted(safe_glyphs):
      left_input.append(routlines[left])
      right_input.append(loutlines[right])
      o_left_input.append(loutlines["o"])
      o_right_input.append(routlines["o"])
      n_left_input.append(loutlines["n"])
      n_right_input.append(routlines["n"])

  predictions = model.predict({
    "left":  np.array(left_input),
    "right": np.array(right_input),
    "left_n": np.array(n_left_input),
    "right_n": np.array(n_right_input),
    "left_o": np.array(o_left_input),
    "right_o": np.array(o_right_input)
    })

  loop = 0
  classes = np.argmax(predictions, axis=1)

  for left in sorted(safe_glyphs):
    for right in sorted(safe_glyphs):
      prediction = classes[loop]
      if bin2label(prediction) != '0':
        print(left, right, bin2label(prediction))
      loop = loop + 1

do_a_font(sys.argv[1])
