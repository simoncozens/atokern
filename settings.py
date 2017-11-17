import glob

training_files = glob.glob("kern-dump/S*.?tf")
validation_files = glob.glob("kern-dump/validation/*.?tf")

# Hyperparameters. These are all guesses.
augmentation = 3
batch_size = 1024
# depth = 10
# width = 8
dropout_rate = 0.3
init_lr = 0.001
lr_decay = 0.5
mu = 0.3
# We predicted 0 but it wasn't
false_negative_penalty = 2
# It was 0 but we said it wasn't
false_positive_penalty = 2

input_names = [
"rightofl", "leftofr",
"rightofn", "leftofo",
"rightofH", "leftofO"
]
regress = False
threeway = False
trust_zeros = True

# Class weights have been precomputed from available data
# This is because we want to feed classes in by generator
# and when we do that, we don't know the weightings in advance
class_weights = [
    2315.52,     13894.953,    32620.591,    29879.511,    17788.56,
    16094.317,    22063.566,    23196.987,    30406.138,    31149.27,
    46542.062,    45089.004,    61497.268,    46647.208,  1036272.,       30751.07,
    31916.427,    18661.843,    12845.056,     6220.076,     6958.4,
     3080.824,     1776.911,     1151.792,      959.052,     1069.903]

def bin_kern3(value):
  if value < -5/800: return 0
  if value > 5/800: return 2
  return 1

def bin_kern(value):
  rw = 800
  if value < -150/rw: return 0
  if value < -100/rw: return 1
  if value < -70/rw: return 2
  if value < -50/rw: return 3
  if value < -45/rw: return 4
  if value < -40/rw: return 5
  if value < -35/rw: return 6
  if value < -30/rw: return 7
  if value < -25/rw: return 8
  if value < -20/rw: return 9
  if value < -15/rw: return 10
  if value < -10/rw: return 11
  if value < -5/rw: return 12
  if value < 0: return 13
  if value == 0: return 14
  if value > 50/rw: return 25
  if value > 45/rw: return 24
  if value > 40/rw: return 23
  if value > 35/rw: return 22
  if value > 30/rw: return 21
  if value > 25/rw: return 20
  if value > 20/rw: return 19
  if value > 15/rw: return 18
  if value > 10/rw: return 17
  if value > 5/rw: return 16
  if value > 0: return 15

if threeway:
  kern_bins = 3
  binfunction = bin_kern3
else:
  kern_bins = 26
  binfunction = bin_kern

