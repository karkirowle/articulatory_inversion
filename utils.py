import numpy as np

class EarlyStopping():

    def __init__(self):

        self.patience = 10
        self.patience_counter = 0
        self.threshold = 1e-5
        self.best_val_loss = np.inf

    def call(self, val_loss):

        if (self.best_val_loss - val_loss) > self.threshold:
            self.patience_counter = 0
            self.best_val_loss = val_loss
        else:
            self.patience_counter += 1

        if self.patience_counter == self.patience:
            return True


