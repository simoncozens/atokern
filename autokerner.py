from sidebearings import safe_glyphs, loadfont, samples, get_m_width
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.decomposition import PCA
from termcolor import colored

import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pickle

path=sys.argv[1]

print(colored("Loading font...","yellow"))

if os.path.isfile(path+".pickle") and os.path.getmtime(path+".pickle") < os.path.getmtime(path):
  os.remove(path+".pickle")

loutlines, routlines, kernpairs, mwidth = loadfont(path, None)
obj = {"loutlines": loutlines, "routlines": routlines, "mwidth": mwidth, "kerndata": None}
if not os.path.isfile(path+".pickle"):
  pickle.dump(obj, open(path+".pickle","wb"))

print(colored("\nStep one: Clustering","cyan"))

def kerngroups(lroutlines, label):
  glyphnames = []
  outlines = []
  eps = 250
  for g in loutlines:
    glyphnames.append(g)
    outlines.append(lroutlines[g])
  while True:
    print("\nClustering %s outlines with tightness %s" % (colored(label,"green"), colored(eps,"green")))
    pca = PCA(n_components=3, svd_solver="full").fit(outlines)
    outlines_pca = pca.transform(outlines)

    outlines = np.array(outlines)
    db = DBSCAN(eps=eps,min_samples=1).fit(outlines)
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('Number of kern groups: %s' % colored(n_clusters_,"green"))
    groups = []
    outliers = []
    for i in range(0,len(set(labels))):
      groups.append([])

    for i in range(0,len(labels)):
      groups[labels[i]].append(glyphnames[i])

    gid=1
    names = []
    for g in groups:
      if len(g) > 1:
        name = "@%s_%i" % (label, gid)
        print(name + "= [%s];" % (" ".join(g)))
        names.append(name)
        gid = gid + 1
      else:
        outliers.append(g[0])
        names.append(g[0])
    print("Ungrouped:", ", ".join(outliers))
    new_eps = input(colored("Try another tightness or return to accept current [%i]? "%eps,"white", attrs=["bold"]))
    if not new_eps:
      return groups, names
    try:
      eps = int(new_eps)
    except Exception as e:
      pass

lgroups, lnames = kerngroups(loutlines, "left")
rgroups, rnames = kerngroups(routlines, "right")

print("Total potential kern pairs: %i" % (len(lgroups) * len(rgroups)))

print(colored("\nStep two: Kerning!","cyan"))
print(colored("Loading kern model...","yellow"))

from sidebearings import safe_glyphs, loadfont, samples, get_m_width
from settings import weight_matrix, input_names
from auxiliary import WeightedCategoricalCrossEntropy
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import keras
import os

np.set_printoptions(precision=3, suppress=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

model = keras.models.load_model("kernmodel.hdf5", custom_objects={'w_categorical_crossentropy': WeightedCategoricalCrossEntropy(weight_matrix)})
print(colored("Let's do this...","yellow"))

def leftcontour(letter):
  return np.array(loutlines[letter])/mwidth
def rightcontour(letter):
  return np.array(routlines[letter])/mwidth

input_tensors = { "pair": [] }
for n in input_names:
  input_tensors[n] = []

for r, rname in zip(rgroups, rnames):
  for l, lname in zip(lgroups, lnames):
    right = r[0]
    left = l[0]
    input_tensors["pair"].append(lname+" "+rname)
    input_tensors["rightofl"].append(rightcontour(left))
    input_tensors["leftofr"].append(leftcontour(right))
    input_tensors["rightofr"].append(rightcontour(right))
    input_tensors["leftofl"].append(leftcontour(left))
    input_tensors["rightofo"].append(rightcontour("o"))
    input_tensors["leftofH"].append(leftcontour("H"))

for n in input_names:
  input_tensors[n] = np.array(input_tensors[n])

print("Predicting...")
predictions = np.array(model.predict(input_tensors))
classes = np.argmax(predictions, axis=1)

def bin_to_label(value, mwidth):
  rw = 800
  scale = mwidth/rw
  if value == 0:
    low = int(-150 * scale); high = int(-100 * scale)
  if value == 1:
    low = int(-100 * scale); high = int(-70 * scale)
  if value == 2:
    low = int(-70 * scale); high = int(-50 * scale)
  if value == 3:
    low = int(-50 * scale); high = int(-45 * scale)
  if value == 4:
    low = int(-45 * scale); high = int(-40 * scale)
  if value == 5:
    low = int(-40 * scale); high = int(-35 * scale)
  if value == 6:
    low = int(-35 * scale); high = int(-30 * scale)
  if value == 7:
    low = int(-30 * scale); high = int(-25 * scale)
  if value == 8:
    low = int(-25 * scale); high = int(-20 * scale)
  if value == 9:
    low = int(-20 * scale); high = int(-15 * scale)
  if value == 10:
    low = int(-15 * scale); high = int(-10 * scale)
  if value == 11:
    low = int(-11 * scale); high = int(-5 * scale)
  if value == 12:
    low = int(-5 * scale); high = int(-0 * scale)
  if value == 13:
    return 0
  if value == 14:
    low = int(0 * scale); high = int(5 * scale)
  if value == 15:
    low = int(5 * scale); high = int(10 * scale)
  if value == 16:
    low = int(10 * scale); high = int(15 * scale)
  if value == 17:
    low = int(15 * scale); high = int(20 * scale)
  if value == 18:
    low = int(20 * scale); high = int(25 * scale)
  if value == 19:
    low = int(25 * scale); high = int(30 * scale)
  if value == 20:
    low = int(30 * scale); high = int(50 * scale)
  return int((low+high)/10)*5

loop = 0
total_pairs = 0
for pair in input_tensors["pair"]:
  prediction = classes[loop]
  units = bin_to_label(prediction,mwidth)
  if units != 0:
    total_pairs = total_pairs + 1
    print("kern %s %i;" % (pair, units))
  loop = loop + 1
print("# Total pairs: %i" % total_pairs)