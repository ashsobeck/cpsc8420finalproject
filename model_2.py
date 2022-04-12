import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_mode_2(num_classes = 10):

    input_size = (32, 32, 3)

    model = keras.Sequential(
        [
            keras.Input(shape=input_size),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.2),

            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.2),

            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.Dropout(0.2),

            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D(pool_size=(2,2)),

            layers.Flatten(),
            layers.Dropout(0.2),
            layers.Dense(1024, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ]
    )
    return model