import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import freetype
import numpy as np
import pickle
import sys
import os.path
import glob

epoch = 0

# Hyperparameters. These are all guesses. (Samples should be OK.)

samples = 201
n_l_inputs = samples
n_r_inputs = n_l_inputs
n_l_hidden = int(n_l_inputs/2)
n_r_hidden = int(n_l_inputs/2)
n_hidden2 = 25
n_outputs = 1
learning_rate = 1e-9

# Design the network:

# Input and output
l_input = tf.placeholder(tf.float32,shape=(None,n_l_inputs), name="left")
r_input = tf.placeholder(tf.float32,shape=(None,n_r_inputs), name="right")
kernvalue = tf.placeholder(tf.float32,shape=(None), name="kernvalue")

# Hidden layers
with tf.name_scope("atokern"):
  l_hidden = fully_connected(l_input, n_l_hidden, scope="l_hidden")
  r_hidden = fully_connected(r_input, n_r_hidden, scope="r_hidden")
  wide_input = tf.concat([l_hidden,r_hidden],1)
  hidden2 = fully_connected(wide_input, n_hidden2, scope="hidden2")
  logit = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
  # logit = tf.Print(logit,[logit], "Prediction: ")

# Loss, training and evaluation operations
with tf.name_scope("loss"):
  loss = tf.reduce_mean(tf.squared_difference(logit, kernvalue)) ## SME
with tf.name_scope("train"):
  optimizer = tf.train.AdamOptimizer(learning_rate)
  training_op = optimizer.minimize(loss)
with tf.name_scope("eval"):
  accuracy = tf.reduce_mean(tf.abs(logit-kernvalue))
tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

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

# Something is O(n^2) about this :(
def unpack_mono_bitmap(bitmap):
  data = bytearray(bitmap.rows * bitmap.width)
  for y in range(bitmap.rows):
    for byte_index in range(bitmap.pitch):
      byte_value = bitmap.buffer[y * bitmap.pitch + byte_index]
      num_bits_done = byte_index * 8
      rowstart = y * bitmap.width + byte_index * 8
      for bit_index in range(min(8, bitmap.width - num_bits_done)):
        bit = byte_value & (1 << (7 - bit_index))
        data[rowstart + bit_index] = 1 if bit else 0
  return data

# Turn a glyph into a tensor of boundary samples
def glyph_to_sb(face, data, which="L"):
  glyph = face.glyph
  sb = []
  w, h = glyph.bitmap.width, glyph.bitmap.rows
  scale = int(face.units_per_EM/face.height*samples)
  ascender = int(face.ascender/face.units_per_EM*scale)+1
  height = int(face.height/face.units_per_EM*scale)+1

  if which == "L":
    iterx = range(w)
    last = w-1
    const = glyph.metrics.horiBearingX / 64.0
  else:
    iterx = range(w-1,-1,-1)
    last = 0
    const = glyph.metrics.horiAdvance / 64.0 - (w + glyph.metrics.horiBearingX / 64.0)

  for _ in range(ascender-int(glyph.metrics.horiBearingY / 64)):
    sb.append(int(const+w) / face.max_advance_width)

  for y in range(h):
    for x in iterx:
      if data[w*y+x] == 1 or x==last:
        # print("*",end="")
        if which == "L":
          sb.append(int(const+x) / face.max_advance_width)
        else:
          sb.append(int(const+(w-x)) / face.max_advance_width)
        break
  while len(sb) < height:
    sb.append(int(const+w) / face.max_advance_width)
  return sb

# Trains the NN given a font and its associated kern dump
def do_a_font(path, kerndump):
  face = freetype.Face(path)
  scale = int(face.units_per_EM/face.height*samples)
  face.set_char_size( 64*scale )
  global epoch

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

  def get_batch_size_one(left,right):
    left_batch = []
    right_batch = []
    kern_batch = []
    left_batch.append(loutlines[left])
    right_batch.append(routlines[right])
    kern_batch.append(kernpairs[left][right])
    return np.array(left_batch), np.array(right_batch), np.array(kern_batch)

  def get_batch(left):
    left_batch = []
    right_batch = []
    kern_batch = []
    for right in safe_glyphs:
      # Get the right outline of the left glyph and vice versa
      left_batch.append(routlines[left])
      right_batch.append(loutlines[right])
      kern_batch.append(kernpairs[left][right])
    return np.array(left_batch), np.array(right_batch), np.array(kern_batch)

  # Train!
  with tf.Session() as sess:
    merged = tf.summary.merge_all()
    init.run()
    if os.path.isfile("atokern.ckpt.meta"):
      saver.restore(sess, "./atokern.ckpt")
    train_writer = tf.summary.FileWriter('train_summaries')

    for left in safe_glyphs:
      epoch = epoch + 1
      left_batch, right_batch, kern_batch = get_batch(left)
      feed = { l_input: left_batch,
        r_input: right_batch,
        kernvalue: kern_batch
      }
      summary,_ = sess.run([merged,training_op], feed_dict = feed)
      acc_train = accuracy.eval(feed_dict = feed)
      train_writer.add_summary(summary, epoch)

    save_path = saver.save(sess, "./atokern.ckpt")

for i in glob.glob("./kern-dump/*.otf"):
  print(i)
  do_a_font(i,i+".kerndump")