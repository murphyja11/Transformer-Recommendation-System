import numpy as np
import tensorflow as tf


class Model():
    def __init__(self):
        # Hyperparameters
        self.batch_size = None
        self.learning_rate = .001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Parameters


    @tf.function
    def call(self, inputs, decoder_input):

        pass

    def loss(self, probs, labels, mask):

        pass

    def accuracy(self, probs, labels, mask):

        pass


