import glob
import argparse
parser = argparse.ArgumentParser()

training_files = { x.replace(".kerndump","") for x in glob.glob("kern-dump/*.kerndump") }
validation_files = glob.glob("kern-dump/validation/*.?tf")
output_path = "output/kernmodel.hdf5"

# Hyperparameters. These are all guesses.
# 128,3,512,0.2 with a 16/2 conv layer after input and 0.2 L2 reg
# and new loss function
# works pretty well but converges very slowly: 64% after 16 epochs 
# Generalization is great, though.

# Wide and shallow (1024/3/2048/0.08 reg) pretty good, 61% after 4.
# Changing to old loss function goes great guns. 128/3/512/0.2, start
# lr 1e-5, no L2reg (yet). 73% after 22 epochs, test about 5% behind, no LR drop
# Can't beat 80% though?
# Dropped batch size, removed conv layer and 1st eep layer over inputs,
# Still not beating 80%.
# Tried reducing penalties to 1; not much difference

# 1024/3/256 with 16/8conv and a 1024->64 triangle gets to 90% in 10, 95% in 30 epochs.
# No conv: 80% in 10.

# Widening to 3/1024 makes it better
# 2/4096 is amazeballs but slow. 95% in 5. Generalizatino poor.
# 3/256 is pretty adequate though; using 64/8 gets to 92% in 10.

parser.add_argument('--batch_size',  nargs='?', type=int, default=1024)
parser.add_argument('--max_epochs',  nargs='?', type=int, default=6000)
parser.add_argument('--depth', nargs='?', type=int, default=3)
parser.add_argument('--width', nargs='?', type=int, default=1024)
parser.add_argument('--dropout_rate', nargs='?', type=float, default=0.1)
parser.add_argument('--init_lr', nargs='?', type=float, default=1e-5)
parser.add_argument('--relu_reg', nargs='?', type=float, default=0.04)
parser.add_argument('--tri_reg', nargs='?', type=float, default=0.04)
parser.add_argument('--max_per_font', nargs='?', type=int, default=2000)
parser.add_argument('--write_batch_performance', action='store_true')

args = parser.parse_args()

augmentation =1
lossfunction = "old"
batch_size = args.batch_size
max_epochs = args.max_epochs
depth = args.depth
width = args.width
dropout_rate = args.dropout_rate
init_lr = args.init_lr
relu_reg = args.relu_reg
tri_reg = args.tri_reg
write_batch_performance = args.write_batch_performance

lr_decay = 0.5
mu = 0.3
# We predicted 0 but it wasn't
false_negative_penalty = 1
# It was 0 but we said it wasn't
false_positive_penalty = 1
all_pairs = True
mirroring = True
max_per_font = args.max_per_font

input_names = [
"rightofl", "leftofr",
#"rightofn", "leftofn",
"rightofo",
#"leftofo",
"leftofH", 
# "leftofH",
#"rightofO", "leftofO",
]
regress = False
threeway = False
trust_zeros = True
covnet = True
generate = False

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

