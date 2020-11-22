import numpy as np
import tensorflow as tf


class Model():
    def __init__(self):
        # Hyperparameters
        self.batch_size = None
        self.learning_rate = .001

        # Parameters


    @tf.function
    def call(self, events, user_info):

        pass

    def loss(self, probs, labels, mask):

        pass

    def accuracy(self, probs, labels, mask):

        pass


