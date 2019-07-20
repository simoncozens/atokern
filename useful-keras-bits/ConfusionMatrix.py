from keras.callbacks import Callback
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import io

class ConfusionMatrix(Callback):
  def __init__(self, tbc, categories, fluid = False):
    super(ConfusionMatrix, self).__init__()
    self.fluid = fluid
    self.categories = categories
    self.tbc = tbc

  def on_epoch_end(self, epoch, logs={}):
    # XXX fix for multiinput/output nets
    x_test = self.model.validation_data[0]
    y_test = np.argmax(self.model.validation_data[1],axis=1)
    y_pred = self.model.predict(x_test)
    if fluid:
      y_true = y_test.ravel()
      histogram = np.zeros((self.categories,self.categories))
      np.add.at(histogram, y_true, y_pred)
    else:
      y_pred = np.argmax(y_pred,axis=1)
      histogram = sklearn.metrics.confusion_matrix(y_test, y_pred)
      histogram = np.log(histogram + 1)
    fig = plt.figure()
    plt.imshow(histogram, cmap='nipy_spectral')#, interpolation='bilinear')
    
    output = io.BytesIO()
    plt.savefig(output, format="png")
    heatmap = tf.Summary.Image(encoded_image_string=output.getvalue(),
                                   height=7,
                                   width=7)
    plt.close()
    summary = tf.Summary(value=[tf.Summary.Value(tag="Confusion",
image=heatmap)])
    self.tbc.get_deep_writers("validation").add_summary(summary, epoch)

