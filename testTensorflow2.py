import tensorflow as tf
from os import path, getcwd, chdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc')>0.999):
            print("\nReached 60% accuracy so cancelling training!")
            self.model.stop_training = True
      
callbacks = myCallback()
mnist = tf.keras.datasets.mnist
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

training_images  = training_images / 255.0
test_images = test_images / 255.0

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1280, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5, callbacks = [callbacks])

classifications = model.predict(test_images)

print(classifications[0])
