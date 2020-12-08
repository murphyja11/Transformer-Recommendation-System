import numpy as np
import tensorflow as tf
from seq2seq_transformer import *


class Model(tf.keras.Model):
    def __init__(self, num_songs):
        super(Model, self).__init__()
        # Hyperparameters
        self.batch_size = 1
        self.learning_rate = .001

        # Candidate Generation Loss function
        self.candgen_loss = tf.keras.losses.CosineSimilarity()

        # Embedding Layers
        self.event_emb_size = 50
        self.event_emb = tf.keras.layers.Embedding(4, self.event_emb_size)
        self.info_emb_size = 50
        self.info_emb = tf.keras.layers.Embedding(3, self.info_emb_size)

        # Candidate Generation Dense Layers
        self.dense1_output = 100
        self.dense2_output = num_songs
        self.dense1 = tf.keras.layers.Dense(self.dense1_output, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.dense2_output)

        # Ranking Transformer Layers
        self.strn = Seq2Seq_Transformer()


    @tf.function
    def call(self, events, user_info):
        """
        This function calls the model
        First, it calculates the embedding for each event and each user

        :param events: an array of user events, of shape [batch size, num events, 4]
        :param user_info: an array of user info, of shape [batch size, 3]
        :return:
        """
        event_embedding = self.event_emb(events) # May consider removing timestamp information from this step
        info_embedding = self.info_emb(user_info)
        events_combined = tf.reduce_mean(event_embedding, -2)
        # ^ average events embeddings across events.  We can try different strategies here
        inputs = tf.concat([events_combined, info_embedding], 1)

        # Now feed embedded and concatenated events to dense layers for Candidate Generation
        output = self.dense1(inputs)
        output = self.dense2(output) # outputs logits, one per song (so it will probably be a ton of songs)

        stransf =  self.strn(output)

        # Now perform ranking using the Transformer layer


    def loss(self, probs, labels, mask):
        return 0.0

    def accuracy(self, probs, labels, mask):
        return 0.5


