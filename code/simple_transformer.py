import tensorflow as tf
import numpy as np
from transformer_funcs import *


class Simple_Transformer:
    def __init__(self):
        # Hyperparameters
        self.batch_size = 250
        self.learning_rate = .001
        self.window_size = 100

        # Event Embedding Layer
        self.event_emb_size = 50
        self.event_emb = tf.keras.layers.Embedding(4, self.event_emb_size)

        # Positional Encoding Layer
        self.positional_enc = Positional_Encoding_Layer(self.window_size, self.event_emb_size)

        # Single Encoder
        self.encoder1 = Transformer_Block(self.event_emb_size, False)
        # Can add more here potentially!!!

        # Dense Layers
        self.dense1_size = 250
        self.dense1 = tf.keras.layers.Dense(self.dense1_size, activation='relu')
        self.dense2_size = 125
        self.dense2 = tf.keras.layers.Dense(self.dense2_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.num_songs, acitvation='softmax')

    @tf.function
    def call(self, inputs):
        """
        Computes the forward pass of the model

        :param inputs: a tensor of shape (batch size, window size)
        :return: a tensor of shape (batch size, window size, num songs)
        """
        # Embedding layers
        embedding = self.event_emb(inputs)
        embedding = self.positional_enc(embedding)

        # Transformer layer(s)
        output = self.encoder1(embedding)

        # Dense layers
        output = self.dense1(output)
        output = self.dense2(output)
        probs = self.dense3(output)

        return probs

    def loss(self, probs, labels, mask):
        """
        Calculates model cross entropy loss after one forward pass

        :param probs: probabilities of each song
        :param labels: prediction label
        :param mask: padding mask
        :return: the loss of the model
        """

        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
        loss = tf.boolean_mask(loss, mask)

        return tf.reduce_sum(loss)

    def accuracy(self, probs, labels, mask):
        """
        Computes the accuracy of the model over a given batch of data

        :param probs: probabilities produced by the forward pass
        :param labels: the correct labels
        :param mask: a mask to prevent the model from looking ahead
        :return: the accuracy a 1d tensor
        """

        predictions = tf.argmax(input=probs, axis=2)
        accuracy = tf.cast(tf.equal(predictions, labels), dtype=tf.float32)
        accuracy = tf.reduce_mean(tf.boolean_mask(accuracy, mask))

        return accuracy