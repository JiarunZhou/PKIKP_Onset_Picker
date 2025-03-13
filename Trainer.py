import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import os, h5py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.losses import Huber, Reduction
from tensorflow.keras.optimizers.legacy import Adam
from keras.callbacks import EarlyStopping
from time import time
from obspy.core import UTCDateTime

def load_training_data(file_addr, loading_num = 1000000):
    if file_addr[-5:] != ".hdf5": file_addr += ".hdf5"

    print("Loading a dataset: %s"%file_addr)
    f = h5py.File(file_addr,"r")
    x = np.array(f["data"])[:loading_num]
    y = np.array(f["t0"])[:loading_num]
    f.close()
    print("Loaded number: %d"%len(x))

    return x,y

def CNN_picker(npts, normalize = True, lr = 0.001):
    n_filters = [32, 64, 128, 256]
    s_kernels = [7, 5, 4, 3]
    
    # CNN
    inputs = Input(shape=(npts,1))
    x = inputs
    for n_filter, s_kernel in zip(n_filters, s_kernels):
        x = Conv1D(filters = n_filter, kernel_size = s_kernel, padding = 'same', activation = "relu")(x)
        x = MaxPooling1D()(x)
        if normalize:
            x = BatchNormalization()(x)
    x = Flatten()(x)
    
    # FCNN
    for _ in range(2):
        x = Dense(200, activation = 'relu')(x)
        if normalize:
            x = BatchNormalization()(x)

    outputs = Dense(1, activation = "linear")(x)
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(loss=Huber(reduction = Reduction.SUM_OVER_BATCH_SIZE),
                  optimizer=Adam(learning_rate = lr))
    
    return model

def plot_hist_curve(hist, save_plot):
    train_loss = hist.history["loss"]
    val_loss = hist.history["val_loss"]
    plt.figure()
    plt.plot(range(len(train_loss)),train_loss, label = "Train_loss")
    plt.plot(range(len(val_loss)),val_loss, label = "Validation_loss")
    plt.xlabel("Epoch")
    plt.legend()
    if save_plot != False:
        plt.savefig(save_plot, dpi = 300)
    plt.show()

def trainer(x, y, model, 
            epochs = 20, 
            validation_split = 0.2, 
            batch_size = 32, 
            early_stop = True, 
            verbose_training = 1, 
            plot_hist = True, save_plot = "Loss_curve.jpg"):

    callbacks = []
    if early_stop == True:
        callbacks.append(
        EarlyStopping(monitor = 'val_loss', 
                    start_from_epoch = 10,
                    patience = 5, 
                    restore_best_weights = True, 
                    verbose = 1)
        )

    start_training_time = time()
    hist = model.fit(x, y, 
                    epochs = epochs, 
                    validation_split = validation_split, 
                    batch_size = batch_size,
                    callbacks = callbacks, 
                    verbose = verbose_training)
    performance_time = time() - start_training_time

    print("Training starts at",UTCDateTime(start_training_time),"; costing:",performance_time)

    if plot_hist == True:
        plot_hist_curve(hist, save_plot)
      
    return hist