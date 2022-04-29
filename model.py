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
from model_2 import make_model_2
import os
from sklearn.metrics import classification_report
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

def eval_model(model, test_imgs, test_labels, file):
    prediction = model.predict(test_imgs)
    prediction = np.argmax(prediction, axis=1)
    file.write(classification_report(y_true=test_labels, y_pred=prediction))
    print(classification_report(y_true=test_labels, y_pred=prediction))
    

def main():
    # hyperparams
    learning_rate = 1e-3
    batch_size = 32
    epochs = 200
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
    plt.savefig('figures/compressed_imgs.png')

    training_data_control = []
    training_data_other = []
    for i in range(len(compress_list)):
        columns = compress_list[i]
        #Takes about 12-15 seconds on my mac to compress just the training images
        if columns < img_size:
            train_imgs = compress_images(train_imgs, columns)
            test_imgs = compress_images(test_imgs, columns)

        small_model = make_control_model()
        #control_model.summary()
        large_model = make_model_2()
        #other_model.summary()

        large_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[
                                    keras.metrics.SparseCategoricalAccuracy()
                            ]
                            )

        
        small_model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), 
                            loss=keras.losses.SparseCategoricalCrossentropy(),
                            metrics=[
                                    keras.metrics.SparseCategoricalAccuracy()
                            ]
                            )
        train = small_model.fit(train_imgs, 
                        train_labels, 
                        epochs=epochs,
                        shuffle=True,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.ModelCheckpoint(f'./checkpoints/model_save_{i}.ckpt', 
                                                            monitor="val_loss",
                                                            mode="max",
                                                            save_freq='epoch',
                                                            save_best_only=True)  
                        ],
                        validation_data=(test_imgs,test_labels)
                        )

        train_2 = large_model.fit(train_imgs, 
                        train_labels, 
                        epochs=epochs,
                        shuffle=True,
                        batch_size=batch_size,
                        callbacks=[
                            keras.callbacks.ModelCheckpoint(f'./checkpoints/large_model_save_{i}.ckpt', 
                                                            monitor="val_loss",
                                                            mode="max",
                                                            save_freq='epoch',
                                                            save_best_only=True)  
                        ],
                        validation_data=(test_imgs,test_labels)
                        )
        training_data_control.append(train)
        training_data_other.append(train_2)
        with open('results.txt', 'a') as results_file:
            results_file.write(f'Small model {i} columns results:\n')
            # small_model.load_weights(f'./checkpoints/model_save_{i}.ckpt')
            # large_model.load_weights(f'./checkpoints/large_model_save_{i}.ckpt')
            eval_model(small_model, test_imgs, test_labels, results_file)
            small_model_results = small_model.evaluate(test_imgs, test_labels)
            large_model_results = large_model.evaluate(test_imgs, test_labels)
            results_file.write(f'{small_model_results}\n')
            results_file.write(f'Large model {i} columns results:\n')
            eval_model(large_model, test_imgs, test_labels, results_file)
            results_file.write(f'{large_model_results}\n')

    #create folder for figures
    os.makedirs("figures", exist_ok=True)
    #typed it all out to ensure I didn't mess it up. Plotting graphs hasn't been fully tested yet haha
    plot_graphs(training_data_control=training_data_control, training_data_other=training_data_other, compress_list=compress_list)

if __name__ == "__main__":
    main()