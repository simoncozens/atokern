import os
import tensorflow as tf
from keras.callbacks import TensorBoard
import numpy as np
from sklearn.preprocessing import scale,normalize
import sklearn.metrics
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

fluid = False

class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

        x_test = self.validation_data[0:4]
        y_test = np.argmax(self.validation_data[4],axis=1)
        y_pred = self.model.predict(x_test)
        if fluid:
            y_true = y_test.ravel()
            histogram = np.zeros((21,21))
            np.add.at(histogram, y_true, y_pred)
        else:
            y_pred = np.argmax(y_pred,axis=1)
            histogram = sklearn.metrics.confusion_matrix(y_test, y_pred)
        histogram = np.log(histogram + 1)
        fig = plt.figure()
        plt.imshow(histogram, cmap='nipy_spectral')#, interpolation='bilinear')
        import io
        output = io.BytesIO()
        plt.savefig(output, format="png")
        heatmap = tf.Summary.Image(encoded_image_string=output.getvalue(),
                                   height=7,
                                   width=7)
        plt.close()
        summary = tf.Summary(value=[tf.Summary.Value(tag="Confusion", 
image=heatmap)])
        self.val_writer.add_summary(summary, global_step=epoch)
        self.val_writer.flush()


    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
