#!/usr/bin/env python
import glob
import string
import random
import numpy as np

pos_shift_range = (40,120)
neg_shift_range = (-120,-40)

from nntools import NetworkTools
from tensorfont import Font,GlyphRendering
import pickle
from tqdm import tqdm

def generator(f,l,r,validation=False):
  variation = 0
  distribution = 0.5
  if validation:
    distribution = 0.3
  if (random.random() < distribution):
    if random.random() > 0.5:
      variation = random.randint(neg_shift_range[0],neg_shift_range[1])
    else:
      variation = random.randint(pos_shift_range[0],pos_shift_range[1])

  s = "OH"+l+r+"HO"
  img = f.set_string(s, { (l,r): f.pair_distance(l,r)+variation*f.scale_factor})
  img = img.with_padding_to_size(160,550)
  if img is None:
    return None, None
  xtrue = { "string_image": np.clip(img,0,1) }
  if variation > 0:
    ytrue = [0,0,1]
  elif variation == 0:
    ytrue = [0,1,0]
  else:
    ytrue = [1,0,0]
  return xtrue, ytrue

training_files = glob.glob("fonts/training/*tf")
validation_files = glob.glob("fonts/validation/*tf")

net = NetworkTools(generator,
  left_glyphs = string.ascii_uppercase,
  right_glyphs = string.ascii_uppercase,
  net_type = "categorizer",
  category_count = 3,
  batch_size = 32,
  validation_files =  validation_files,
  training_files =  training_files
)

def write_data(files,filename,validation=False):
  inputs = {}
  outputs = []
  ix = 0
  with tqdm(total=len(files)*len(net.left_glyphs)*len(net.right_glyphs)) as pbar:
    for fn in files:
      f = Font(fn,net.font_xheight)
      for l in net.left_glyphs:
        for r in net.right_glyphs:
          try:
            xtrue, ytrue = generator(f,l,r,validation=validation)
            if not (xtrue is None):
              inputs["ix-%05i" % ix] = xtrue["string_image"] > 0.5
              ix = ix + 1
              outputs.append(ytrue)
          except ValueError:
            pass
          pbar.update(1)

  indices = range(0,len(inputs))
  np.savez_compressed("out-"+filename, outputs=outputs)
  np.savez_compressed("in-"+filename, **inputs)

# write_data(training_files,"training")
# write_data(validation_files,"validation",validation=True)

t_inputs = np.load("in-training.npz")
t_outputs = np.load("out-training.npz")["outputs"]
t_indices = []

v_inputs = np.load("in-validation.npz")
v_outputs = np.load("out-validation.npz")["outputs"]
v_indices = []

def tgenerator(f,l,r,validation=False):
  global t_indices
  if len(t_indices) == 0:
    t_indices = list(range(0,len(t_outputs)))
    random.shuffle(t_indices)
  ix = t_indices.pop()
  input_x = t_inputs["ix-%05i" % ix]
  input_x = GlyphRendering.init_from_numpy(None,input_x)
  return { "string_image": input_x }, t_outputs[ix]

def vgenerator(f,l,r,validation=False):
  global v_indices
  if len(v_indices) == 0:
    v_indices = list(range(0,len(v_outputs)))
    random.shuffle(v_indices)
  ix = v_indices.pop()
  input_x = v_inputs["ix-%05i" % ix]
  input_x = GlyphRendering.init_from_numpy(None,input_x)
  return { "string_image": input_x }, v_outputs[ix]

net.generator = net.make_generator(tgenerator,[training_files[0]])
net.validation_generator = net.make_generator(vgenerator,[validation_files[0]])
net.build_network(init_lr=0.5e-3)
net.train(steps_per_epoch = int(len(t_outputs) / net.batch_size), validation_steps = int(len(v_outputs) /net.batch_size))
