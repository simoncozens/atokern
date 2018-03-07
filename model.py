import numpy as np
from keras.layers import Input, Embedding, LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, BatchNormalization, Activation
from keras.models import Model
from keras.constraints import maxnorm
from keras import regularizers
from keras import backend as K
from auxiliary import bigram_frequency, mse_penalizing_miss, create_class_weight, hinged_min_error
from SignalHandler import SignalHandler
from sidebearings import samples
import keras
import os
from settings import output_path, generate, mirroring, covnet, augmentation, batch_size, dropout_rate, init_lr, lr_decay, input_names, regress, threeway, trust_zeros, mu, binfunction, kern_bins, training_files, validation_files, all_pairs, width, depth, relu_reg, tri_reg, write_batch_performance
np.set_printoptions(precision=3, suppress=False)

# Design the network:
def drop(x): return Dropout(dropout_rate)(x)
def relu(x, layers=1, nodes=32):
  for _ in range(0,layers):
    x = Dense(nodes,kernel_initializer="normal",kernel_regularizer=regularizers.l2(relu_reg))(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)
    x = drop(x)
  return x

print("Building network")

inputs = []
nets = []

for n in input_names:
  if covnet:
    input_ = Input(shape=(samples,1), dtype='float32', name=n)
  else:
    input_ = Input(shape=(samples,), dtype='float32', name=n)
  inputs.append(input_)
  if covnet:
    conv = Activation("relu")(Conv1D(64,4)(input_))
    #maxp = MaxPooling1D(pool_size=2)(conv)
    flat = Flatten()(conv)
    net = flat
  else:
    net = input_
    #net = relu(net,layers=1,nodes=1024)
  nets.append(net)

x = keras.layers.concatenate(nets)
x = relu(x, layers=depth,nodes=width)
x = Dense(512,kernel_initializer="normal",kernel_regularizer=regularizers.l2(tri_reg))(x)
x = Dense(256,kernel_initializer="normal",kernel_regularizer=regularizers.l2(tri_reg))(x)
x = Dense(128,kernel_initializer="normal",kernel_regularizer=regularizers.l2(tri_reg))(x)
x = Dense(64,kernel_initializer="normal",kernel_regularizer=regularizers.l2(tri_reg))(x)

if regress:
  kernvalue = Dense(1, activation="linear")(x)
else:
  kernvalue =  Dense(kern_bins, activation='softmax')(x)

if os.path.exists(output_path):
  model = keras.models.load_model(output_path, custom_objects={'hinged_min_error': hinged_min_error, 'mse_penalizing_miss': mse_penalizing_miss})
else:
  model = Model(inputs=inputs, outputs=[kernvalue])

  print("Compiling network")

  opt = keras.optimizers.adam(lr=init_lr)
  #opt = optimizers.SGD(lr=init_lr, decay=1e-6, momentum=0.9, nesterov=True)


  if regress:
    loss = 'mean_squared_error'
    metrics = []
  else:
    #loss = 'categorical_crossentropy'
    loss = mse_penalizing_miss
    metrics = ['accuracy']
  model.compile(loss=loss, metrics=metrics, optimizer=opt)

print(model.summary())
# Trains the NN given a font and its associated kern dump

checkpointer = keras.callbacks.ModelCheckpoint(filepath='output/kernmodel-cp-val.hdf5', verbose=0, save_best_only=True, monitor="val_loss")
checkpointer2 = keras.callbacks.ModelCheckpoint(filepath='output/kernmodel-cp-loss.hdf5', verbose=0, save_best_only=True, monitor="val_loss")
earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=15, verbose=1, mode='auto')
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_decay, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=1, min_lr=0)
if write_batch_performance:
  tensorboard = keras.callbacks.TensorBoard(log_dir='output/atokern', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, write_batch_performance=True)
else:
  tensorboard = keras.callbacks.TensorBoard(log_dir='output/atokern', histogram_freq=0, batch_size=batch_size, write_graph=False, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

signalhandler = SignalHandler()

callback_list = [
  earlystop,
  checkpointer,
  checkpointer2,
  reduce_lr,
  tensorboard,
  signalhandler
]
