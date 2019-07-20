from keras import backend as K
from keras.callbacks import Callback
import signal
class SignalHandler(Callback):
    def __init__(self, dropfactor = 0.1):
        super(SignalHandler, self).__init__()
        self.signal_received = False
        self.sigusr_received = False
        self.sigsave_received = False
        self.dropfactor = dropfactor

        def time_to_quit(sig, frame):
            self.signal_received = True
            print('\nStopping at end of this epoch')
        def drop_lr(sig,frame):
            self.sigusr_received = True
            print('\nDropping LR at end of this epoch')
        def save_now(sig,frame):
            print('\nSaving model')
            self.model.save("keras-emergency-save.hdf5")

        signal.signal(signal.SIGINT, time_to_quit)
        signal.signal(signal.SIGUSR1, drop_lr)
        signal.signal(signal.SIGUSR2, save_now)
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs={}):
        if self.signal_received:
            self.stopped_epoch = epoch
            self.model.stop_training = True
        if self.sigusr_received:
            old_lr = float(K.get_value(self.model.optimizer.lr))
            new_lr = self.dropfactor * old_lr
            print("\nDropping LR due to sigusr1. LR now %f" % new_lr)
            K.set_value(self.model.optimizer.lr, new_lr)
            self.sigusr_received = False

    def on_train_end(self, logs={}):
        if self.stopped_epoch > 0:
            self.model.save("keras-interrupt-save.hdf5")
            print('Epoch %05d: stopping due to signal' % (self.stopped_epoch))
