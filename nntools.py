import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Flatten, BatchNormalization, Activation, concatenate, Conv2D, MaxPooling2D
# from tensorflow.keras.layers.noise import GaussianNoise
from tensorflow.keras.models import Model
from tensorfont import Font,safe_glyphs_l,safe_glyphs_r,safe_glyphs,GlyphRendering
import random
import numpy as np

# NN helpers
def drop(x): return Dropout(dropout_rate)(x)
def relu(x, layers=1, nodes=32,name=""):
  for _ in range(0,layers):
    x = Dense(nodes, kernel_initializer='he_normal',
                # kernel_constraint=max_norm(3.),
                # kernel_regularizer=regularizers.l2(relu_reg),
                # activity_regularizer=regularizers.l1(relu_reg)
              )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    # x = drop(x)
  return x

class NetworkTools:
  def __init__(self,
    generator,
    net_type = "categorizer",
    category_count = None,
    output_activation = None,
    training_files = None,
    validation_files = None,
    left_glyphs = safe_glyphs_l,
    right_glyphs = safe_glyphs_r,
    batch_size = 32,
    font_xheight = 50,
    box_height = 160,
    box_width = 550):
    self.net_type = net_type
    self.output_activation = output_activation
    self.training_files = training_files
    self.validation_files = validation_files
    self.left_glyphs = left_glyphs
    self.right_glyphs = right_glyphs
    self.batch_size = batch_size
    if len(self.training_files) < 1 or len(self.validation_files) < 1:
      raise ValueError
    self.generator = self.make_generator(generator,training_files)
    self.validation_generator = self.make_generator(generator,validation_files)

    self.font_xheight = font_xheight
    self.box_height = box_height
    self.box_width = box_width
    # Sniff generator to determine input names
    inputs, _ = self.test_generator()
    self.input_names = inputs.keys()

  def test_generator(self):
    self.sniffing = True
    inputs, out = next(self.generator)
    self.sniffing = False
    return inputs, out

  def build_network(self, depth=2, width=32, init_lr = 1e-4):
    inputs = []
    nets = []
    input_t = {}
    conv_filters = 16
    kernel_size = (2, 2)
    pool_size = 2

    for n in self.input_names:
      if "_1d" in n:
        input_ = Input(shape=(1,), name=n)
        inputs.append(input_)
      elif "_image" in n:
        input_ = Input(shape=(self.box_height,self.box_width,1,), name=n)
        inputs.append(input_)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation='relu', kernel_initializer='he_normal',
                       name=n+'_conv1')(input_)
        inner = MaxPooling2D(pool_size=(pool_size, pool_size), name=n+'_max1')(inner)
        inner = BatchNormalization()(inner)
        inner = Conv2D(conv_filters, kernel_size, padding='same',
                       activation='relu', kernel_initializer='he_normal',
                       name=n+'_conv2')(inner)
        input_ = MaxPooling2D(pool_size=(pool_size, pool_size), name=n+'_max2')(inner)
        input_ = BatchNormalization()(input_)
        input_ = Flatten()(input_)
      input_t[n] = input_

    nets = [ input_t[n] for n in self.input_names if not "_exclude" in n]
    if len(nets) == 1:
      concat = nets[0]
    else:
      concat = concatenate(nets)

    concat = relu(concat, layers=depth,nodes=width)
    metrics = []

    if self.net_type == "discriminator":
      output = Dense(1, kernel_initializer='normal', activation='sigmoid')(concat)
      loss = "binary_crossentropy"
      metrics.append("accuracy")
    elif self.net_type == "regression":
      loss = "mse"
      output = Dense(1, kernel_initializer='zeros')(concat)
      if self.output_activation is not None:
        output = Activation(self.output_activation)(output)
    else: # We're a categorizer
      if self.category_count is None:
        raise ValueError
      output = Dense(self.category_count, kernel_initializer='normal', activation='softmax')(concat)
      loss = "categorical_crossentropy"
      metrics.append("accuracy")

    model = Model(inputs=inputs, outputs=[output])
    opt = tf.keras.optimizers.Adam(lr=init_lr)
    model.compile(loss=loss, metrics=metrics, optimizer=opt)
    self.model = model

  def make_generator(self, func, filenames):
    fontobjects = [ Font(fn,self.font_xheight) for fn in filenames ]
    while True:
      if self.sniffing:
        batch_size = 1
      else:
        batch_size = self.batch_size
      input_tensors = {}
      outputs = []
      while len(outputs) < batch_size:
        f = random.choice(fontobjects)
        l = random.choice(self.left_glyphs)
        r = random.choice(self.right_glyphs)
        try:
          x_true, y_true = func(f,l,r)
          if x_true is None:
            raise ValueError
          for n in x_true.keys():
            if "_image" in n:
              x_true[n] = x_true[n].with_padding_to_size(self.box_height,self.box_width)
              if x_true[n] is None:
                raise ValueError
              x_true[n] = x_true[n].reshape((self.box_height,self.box_width,1))
            if not n in input_tensors:
              input_tensors[n] = []
            input_tensors[n].append(x_true[n])
          outputs.append(y_true)
        except ValueError:
          pass
      for k, v in input_tensors.items():
        input_tensors[k] = np.array(v)
      outputs = np.array(outputs)
      yield(input_tensors, outputs)

  def train(self, output_dir=".", epochs=3000, steps_per_epoch = 50, validation_steps = 10,lr_decay=0.5):
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath=output_dir+'/output/kernmodel-cp-val.hdf5', verbose=0, save_best_only=True, monitor="val_loss")
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, mode='auto')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=lr_decay, patience=20, verbose=1, mode='auto', min_delta=1e-6, cooldown=100, min_lr=0)
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=output_dir+"/output", histogram_freq=0,
    write_graph=False, write_grads=False,batch_size=self.batch_size,update_freq='batch',
    write_images=False)

    callback_list = [
      earlystop,
      checkpointer,
      reduce_lr,
      tensorboard
    ]
    self.model.summary()
    print("Training")
    history = self.model.fit_generator(
      generator = self.generator,
      validation_data = self.validation_generator,
      steps_per_epoch = steps_per_epoch,
      validation_steps = validation_steps,
      epochs=epochs, verbose=1, callbacks=callback_list,
      max_queue_size=1
    )
    self.model.save(output_dir+"/output/kernmodel-final.hdf5")


