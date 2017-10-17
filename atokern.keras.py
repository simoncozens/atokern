import freetype
import numpy as np
import pickle
import sys
import os.path
import glob

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
lshape = Dense(32, activation='relu')(lshape)
r_input = Input(shape=(n_r_inputs,), dtype='float32', name='right')
rshape = Dense(64, activation='relu')(r_input)
rshape = Dense(32, activation='relu')(rshape)

x = keras.layers.concatenate([lshape, rshape])
x = Dense(64, activation='relu')(x)
x = Dense(16, activation='relu')(x)

kernvalue =  Dense(1, activation='linear', name='kernvalue')(x)

model = Model(inputs=[l_input, r_input], outputs=[kernvalue])
print("Compiling network")

opt = keras.optimizers.adam()
# opt = keras.optimizers.SGD(lr=0.0001, clipnorm=1.)

model.compile(loss='mean_squared_error',
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
def do_a_font(path, kerndump):
  face = freetype.Face(path)
  face.set_char_size( 64*face.units_per_EM)
  global epoch
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

  def get_batch(left):
    left_batch = []
    right_batch = []
    kern_batch = []
    for right in safe_glyphs:
      # Get the right outline of the left glyph and vice versa
      left_batch.append(routlines[left])
      right_batch.append(loutlines[right])
      kern_batch.append(kernpairs[left][right] / face.max_advance_width)
      # Generate more data by shrinking the sidebearings and
      # correspondingly expanding the kern value
      for shuffle in range(-10,10):
        left_batch.append(routlines[left]-shuffle)
        right_batch.append(loutlines[right])
        kern_batch.append(kernpairs[left][right]+shuffle  / face.max_advance_width)

        left_batch.append(routlines[left])
        right_batch.append(loutlines[right]-shuffle)
        kern_batch.append(kernpairs[left][right]+shuffle  / face.max_advance_width)

    return np.array(left_batch), np.array(right_batch), np.array(kern_batch)

  # Train!
  for left in safe_glyphs:
    epoch = epoch + 1
    left_batch, right_batch, kern_batch = get_batch(left)
    loss = model.train_on_batch([left_batch, right_batch], [kern_batch])
    print(left, "Epoch", epoch, model.metrics_names[0], "=", loss[0], "-", model.metrics_names[1], "=", loss[1])

for i in glob.glob("./kern-dump/*.otf"):
  print(i, end=" ")
  do_a_font(i,i+".kerndump")

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
