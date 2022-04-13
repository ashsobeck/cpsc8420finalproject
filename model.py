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
from image_compression import compress_images
import numpy as np
from model_2 import make_mode_2
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
    
    (train_imgs, train_labels), (test_imgs, test_labels) = keras.datasets.cifar10.load_data()

    #only do first image if you want to just test the compression and compare the images
    '''x = compress_images(train_imgs[0].reshape(1, 32, 32, 3), 0.90)
    plt.imshow(x[0])
    plt.imshow(train_imgs[0])'''

    #Takes about 12-15 seconds on my mac to compress just the training images
    compressed_images = compress_images(train_imgs, 0.90)

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

    # plot accuracy and lost over training time
    plt.plot(train.history['accuracy'])
    plt.plot(train.history['val_accuracy'])
    plt.title('Accuracy Over Time')
    plt.ylabel('accuracy')
    plt.xlabel('epochs')
    plt.legend(['train accuracy','test accuracy'])
    plt.savefig('./acc_over_time.png')
    plt.show()
    
    plt.plot(train.history['loss'])
    plt.plot(train.history['val_loss'])
    plt.title('Loss Over Time')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train loss','test loss'])
    plt.savefig('./loss_over_time.png')
    plt.show()

if __name__ == "__main__":
    main()