#!/usr/bin/env python
import glob
import string
import random
import numpy as np

pos_shift_range = (50,120)
neg_shift_range = (-120,-50)

from nntools import NetworkTools

def generator(f,l,r):
  variation = 0
  if random.random() < 0.5:
    if random.random() > 0.5:
      variation = random.randint(neg_shift_range[0],neg_shift_range[1])
    else:
      variation = random.randint(pos_shift_range[0],pos_shift_range[1])

  s = "OH"+l+r+"HO"
  img = f.set_string(s, { (l,r): f.pair_distance(l,r)+variation*f.scale_factor})
  xtrue = { "string_image": img.transform(lambda x: np.clip(x,0,1)) }
  if variation == 0:
    ytrue = 1
  else:
    ytrue = 0
  return xtrue, ytrue

net = NetworkTools(generator,
  left_glyphs = string.ascii_uppercase,
  right_glyphs = string.ascii_uppercase,
  net_type = "discriminator",
  validation_files =  glob.glob("fonts/validation/*tf"),
  training_files =  glob.glob("fonts/training/*tf")
)

net.build_network()
net.train()