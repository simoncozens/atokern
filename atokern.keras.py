import freetype
import numpy as np
import pickle
import sys
import os.path
import glob
import random
from math import copysign

from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
import keras

epoch = 0

# Hyperparameters. These are all guesses. (Samples should be OK.)

samples = 100
n_l_inputs = samples
n_r_inputs = n_l_inputs

# Design the network:
print("Building network")
# Input and output
l_input = Input(shape=(n_l_inputs,), dtype='float32', name='left')
lshape = Dense(64, activation='relu')(l_input)
r_input = Input(shape=(n_r_inputs,), dtype='float32', name='right')
rshape = Dense(64, activation='relu')(r_input)

n_left_input = Input(shape=(samples,), dtype='float32', name='left_n')
n_right_input = Input(shape=(samples,), dtype='float32', name='right_n')
nlshape = Dense(64, activation='relu')(n_left_input)
nrshape = Dense(64, activation='relu')(n_right_input)

o_left_input = Input(shape=(samples,), dtype='float32', name='left_o')
o_right_input = Input(shape=(samples,), dtype='float32', name='right_o')
olshape = Dense(64, activation='relu')(o_left_input)
orshape = Dense(64, activation='relu')(o_right_input)

x = keras.layers.concatenate([lshape, rshape, nlshape, nrshape, olshape, orshape])
x = Dense(64, activation='relu')(x)
x = Dense(16, activation='relu')(x)

kern_bins = 7

kernvalue =  Dense(kern_bins, activation='softmax')(x)

def bin_kern(value):
  if value < -20: return 1
  if value < -10: return 2
  if value < 0: return 3
  if value == 0: return 4
  if value > 0: return 5
  if value > 10: return 6
  if value > 20: return 7

model = Model(inputs=[l_input, r_input, n_left_input, n_right_input, o_left_input, o_right_input], outputs=[kernvalue])
print("Compiling network")

opt = keras.optimizers.adam()
# opt = keras.optimizers.SGD(lr=0.0001, clipnorm=1.)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


# I figure it's only worth training examples that we can be
# relatively sure are well kerned - basic Latin letters, numbers and
# punctuation

#safe_glyphs = set()
#for i in range(1,face.num_glyphs):
#  safe_glyphs.add(face.get_glyph_name(i).decode())

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
    newsb.append(sb[sliceval] / face.max_advance_width)
  return newsb

# Trains the NN given a font and its associated kern dump
def do_a_font(path, kerndump, epoch):
  face = freetype.Face(path)
  face.set_char_size( 64*face.units_per_EM)
  loutlines = dict()
  routlines = dict()
  kernpairs = dict()
  def load_kernpairs(file):
    with open(file) as f:
      for line in f:
        l,r,k = line.split()
        if not l in kernpairs:
          kernpairs[l] = dict()
        kernpairs[l][r] = int(k)

  if os.path.isfile(path+".pickle"):
    obj = pickle.load(open(path+".pickle","rb"))
    loutlines, routlines, kernpairs = obj["loutlines"], obj["routlines"], obj["kerndata"]
  else:
    for l in safe_glyphs:
      kernpairs[l]=dict()
      for r in safe_glyphs:
        kernpairs[l][r] = 0

    load_kernpairs(kerndump)
    for g in safe_glyphs:
      glyphindex = face.get_name_index(g.encode("utf8"))
      if glyphindex:
        face.load_glyph(glyphindex, freetype.FT_LOAD_RENDER |
                                  freetype.FT_LOAD_TARGET_MONO)
        data = unpack_mono_bitmap(face.glyph.bitmap)
        loutlines[g] = np.array(glyph_to_sb(face, data, which="L"))
        routlines[g] = np.array(glyph_to_sb(face, data, which="R"))

    obj = {"loutlines": loutlines, "routlines": routlines, "kerndata": kernpairs}
    pickle.dump(obj, open(path+".pickle","wb"))

  left_input = []
  right_input = []
  kern_input = []
  o_left_input = []
  o_right_input = []
  n_left_input = []
  n_right_input = []

  for left in safe_glyphs:
    for right in safe_glyphs:
      left_input.append(routlines[left])
      right_input.append(loutlines[right])
      o_left_input.append(loutlines["o"])
      o_right_input.append(routlines["o"])
      n_left_input.append(loutlines["n"])
      n_right_input.append(routlines["n"])
      kern_input.append(bin_kern(kernpairs[left][right]))

  kerncats = keras.utils.to_categorical(kern_input, num_classes=kern_bins)

  model.fit({
    "left":  np.array(left_input),
    "right": np.array(right_input),
    "left_n": np.array(n_left_input),
    "right_n": np.array(n_right_input),
    "left_o": np.array(o_left_input),
    "right_o": np.array(o_right_input)
    }, kerncats,
    batch_size=32, epochs=2, verbose=1, callbacks=None,
    validation_split=0.2, initial_epoch=0)

files = glob.glob("./kern-dump/*.*tf")
epochn = 0
for _ in range(50):
  random.shuffle(files)
  for i in files:
    print(i)
    epochn = epochn + 5
    do_a_font(i,i+".kerndump", epochn)

  # face = freetype.Face(i)
  # face.set_char_size( 64 *face.units_per_EM )
  # glyphindex = face.get_name_index("g")
  # face.load_glyph(glyphindex, freetype.FT_LOAD_RENDER |
  #                           freetype.FT_LOAD_TARGET_MONO)
  # data = unpack_mono_bitmap(face.glyph.bitmap)
  # print("Left:")
  # print(np.array(glyph_to_sb(face, data, which="L")))
  # print("Right:")
  # print(np.array(glyph_to_sb(face, data, which="R")))
