# -*- coding: utf-8 -*-

"""
Authors: Ashton Sobeck and Will Sherrer
CPSC 8420
"""

import matplotlib.pyplot as plt
import ssl
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from image_compression import compress_images, plot_graphs
import numpy as np
from model_2 import make_mode_2
import os
# ssl._create_default_https_context = ssl._create_unverified_context

def make_control_model():
    # can take any height or width
    input_img = keras.Input(shape=(None, None, 3))
    
    model = layers.Conv2D(filters=32, kernel_size=2, activation="relu")(input_img)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(filters=64, kernel_size=2, activation="relu")(model)
    model = layers.MaxPool2D()(model)
    model = layers.Conv2D(filters=128, kernel_size=2, activation="relu")(model)
    model = layers.MaxPool2D()(model)
    
    model = layers.GlobalAveragePooling2D()(model)
    
    model = layers.Dense(256, activation="relu")(model)
    output = layers.Dense(10, activation="softmax")(model)
    
    full_model = keras.Model(inputs=input_img, outputs=output)
    return full_model

def main():
    # hyperparams
    learning_rate = 1e-3
    batch_size = 32
    epochs = 100
    img_size = 32
    
    (train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.cifar10.load_data()

    #list of columns to be used. 32 for all columns (0% compression), 1 for least columns
    compress_list = [32, 6, 4, 2, 1]

    #this block of code is just to show a test img real quick if you want to with all the compressed images
    rows, col = 2, 3
    for i in range(len(compress_list)):
        x = compress_images(train_imgs[4].reshape(1, img_size, img_size, 3), compress_list[i])
        plt.subplot(rows, col, i + 1)
        #can print columns or percentage in same compress list
        if compress_list[i] >= 1:
            plt.title("%s columns" % (compress_list[i]))
        else:
            plt.title("%s%%" % (compress_list[i]))
        plt.imshow(x[0])
    #since it prints after each compress images, just clean up the terminal
    os.system("clear")
    plt.show()

    training_data_control = []
    training_data_other = []
    for i in range(len(compress_list)):
        columns = compress_list[i]
        #Takes about 12-15 seconds on my mac to compress just the training images
        if columns < img_size:
            train_imgs = compress_images(train_imgs, columns)
            test_imgs = compress_images(test_imgs, columns)

        control_model = make_control_model()
        #control_model.summary()
        other_model = make_mode_2()
        #other_model.summary()

        other_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[keras.metrics.SparseCategoricalAccuracy()]
                            )

        
        control_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[keras.metrics.SparseCategoricalAccuracy()]
                            )
        train = control_model.fit(train_imgs, 
                        train_labels, 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.ModelCheckpoint('./model_save.ckpt', 
                                                            monitor="val_loss",
                                                            mode="max",
                                                            save_freq='epoch',
                                                            save_best_only=True)  
                        ],
                        validation_data=(test_imgs,test_labels)
                        )

        train_2 = other_model.fit(train_imgs, 
                        train_labels, 
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.ModelCheckpoint('./model_save.ckpt', 
                                                            monitor="val_loss",
                                                            mode="max",
                                                            save_freq='epoch',
                                                            save_best_only=True)  
                        ],
                        validation_data=(test_imgs,test_labels)
                        )
        training_data_control.append(train)
        training_data_other.append(train_2)

    #create folder for figures
    os.makedirs("figures", exist_ok=True)
    #typed it all out to ensure I didn't mess it up. Plotting graphs hasn't been fully tested yet haha
    plot_graphs(training_data_control=training_data_control, training_data_other=training_data_other, compress_list=compress_list)

if __name__ == "__main__":
    main()