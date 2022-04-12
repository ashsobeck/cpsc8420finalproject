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
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

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
    control_model = make_control_model()
    
    control_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                          loss=keras.losses.SparseCategoricalCrossentropy(),
                          metrics=[keras.metrics.SparseCategoricalAccuracy()]
                          )
    control_model.fit(train_imgs, 
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

if __name__ == "__main__":
    main()