import numpy as np
import tensorflow as tf
from transformer_funcs import *


class Seq2Seq_Transformer(tf.keras.Model):
    def __init__(self):
        super(Seq2Seq_Transformer, self).__init__()
        # Hyperparameters
        self.batch_size = 250
        self.learning_rate = .001
        self.window_size = 1000000 #100
        self.num_songs = 500

        # Event Embedding Layer
        self.event_emb_size = 1000000 #50
        self.event_emb = tf.keras.layers.Embedding(4, self.event_emb_size)

        # Positional Encoding Layer
        self.positional_enc = Positional_Encoding_Layer(self.window_size, self.event_emb_size)

        # Encoders
        self.encoder1 = Transformer_Block(self.event_emb_size, False, False)
        self.encoder2 = Transformer_Block(self.event_emb_size, False, False)
        self.encoder3 = Transformer_Block(self.event_emb_size, False, False)

        # Decoders
        self.decoder1 = Transformer_Block(self.event_emb_size, True, False)
        self.decoder2 = Transformer_Block(self.event_emb_size, True, False)
        self.decoder3 = Transformer_Block(self.event_emb_size, True, False)

        # Dense Layers
        self.dense1_size = 250
        self.dense1 = tf.keras.layers.Dense(self.dense1_size, activation='relu')
        self.dense2_size = 125
        self.dense2 = tf.keras.layers.Dense(self.dense2_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.num_songs, activation='softmax')

    @tf.function
    def call(self, inputs):
        """
        Computes the models transformers forward pass using the input sequence data

        :param inputs: users listening history (batch size, window size)
        :return: predicted songs at each time step (batch size, window size)
        """
        embeddings = self.event_emb(inputs)
        embeddings = self.positional_enc(embeddings)

        encoded = self.encoder1(embeddings)
        encoded = self.encoder2(encoded)
        encoded = self.encoded3(encoded)

        decoded = self.decoder(encoded)  # TODO: Removed Context?
        # TODO: call Decoder 2 and 3

        output = self.dense1(decoded)
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
