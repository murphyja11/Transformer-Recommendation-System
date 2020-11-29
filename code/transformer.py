import numpy as np
import tensorflow as tf
from transformer_funcs import *

class Transformer():
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

        # Encoders
        self.encoder1 = Transformer_Block(self.event_emb_size, False)
        self.encoder2 = Transformer_Block(self.event_emb_size, False)
        self.encoder3 = Transformer_Block(self.event_emb_size, False)

        # Decoders
        self.decoder1 = Transformer_Block(self.event_emb_size, True)
        self.decoder2 = Transformer_Block(self.event_emb_size, True)
        self.decoder3 = Transformer_Block(self.event_emb_size, True)

        # Dense Layers
        self.dense1_size = 250
        self.dense1 = tf.keras.layers.Dense(self.dense1_size, activation='relu')
        self.dense2_size = 125
        self.dense2 = tf.keras.layers.Dense(self.dense2_size, activation='relu')
        self.dense3 = tf.keras.layers.Dense(self.num_songs, acitvation='softmax')


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

        decoded = self.decoder(encoded, context=encoded) # TODO: <<<<WRONG
        #TODO: call Decoder 2 and 3

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
