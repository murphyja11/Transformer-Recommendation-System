import numpy as np
import tensorflow as tf


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_size, is_decoder, multi_headed=False):
        super(Transformer_Block, self).__init__()

        self.dense1 = tf.keras.layers.Dense(emb_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(emb_size)

        if not multi_headed:
            self.self_attention = Attention_Head(emb_size, emb_size, use_mask=is_decoder)
            if self.is_decoder:
                self.context_attention = Attention_Head(emb_size, emb_size, use_mask=False)
        else:
            self.self_attention = Multi_Headed(emb_size, use_mask=is_decoder)
            if self.is_decoder:
                self.context_attention = Multi_Headed(emb_size, use_mask=False)

        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)

    @tf.function
    def call(self, inputs, context=None):
        """
        This function calls a forward pass on a transformer block

        :param inputs: input sequence tensor (batch size, window size, embedding size)
        :return: a tensor of predictions for each time step (batch size, window size, embedding size)
        """
        attention_out = self.self._attention(inputs, inputs, inputs)
        attention_out += inputs
        attention_norm = self.layer_norm(attention_out)

        if self.is_decoder:
            assert context is not None
            context_attention_out = self.context_attention(context, context, attention_norm)
            context_attention_out += attention_norm
            attention_norm = self.layer_norm(context_attention_out)

        output = self.dense1(attention_norm)
        output += attention_norm
        output = self.layer_norm(output)
        output = tf.nn.relu(output)

        return output


class Positional_Encoding_Layer(tf.keras.layers.Layer):
    def __init__(self, window_size, embedding_size):
        super(Positional_Encoding_Layer, self).__init__()
        # TODO: Is this right??? Or should we calculate using the formula???
        self.embeddings = self.add_weight("pos_embed", shape=[window_size, embedding_size])

    @tf.function
    def call(self, x):
        """
        Add positinal embeddings to word embeddings

        :param x: word embeddings, (batch size, window size, embedding size)
        :return: new word embeddings with positional encodings added
        """
        return x + self.embeddings


class Attention_Head(tf.keras.layers.Layer):
    def __init__(self, input_size, output_size, use_mask):
        super(Attention_Head, self).__init__()

        self.use_mask = use_mask

        self.keys = self.add_weight(name='K', shape=(input_size, output_size), dtype=tf.float32, trainable=True)
        self.values = self.add_weight(name='V', shape=(input_size, output_size), dtype=tf.float32, trainable=True)
        self.queries = self.add_weight(name='Q', shape=(input_size, output_size), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, key_inputs, value_inputs, query_inputs):
        K = tf.tensordot(key_inputs, self.keys, ((2), (0)))
        V = tf.tensordot(value_inputs, self.values, ((2), (0)))
        Q = tf.tensordot(query_inputs, self.queries, ((2), (0)))

        attention_matrix = Attention_Matrix(K, Q, use_mask=self.user_mask)
        Z = tf.matmul(attention_matrix, V)

        return Z


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, size, use_mask):
        super(Multi_Headed, self).__init__()

        self.use_mask = use_mask

    @tf.function
    def call(self, key_inputs, value_inputs, query_inputs):
        pass


def Attention_Matrix(keys, queries, use_mask=False):
    query_window_sz = queries.get_shape()[1]
    key_window_sz = keys.get_shape()[1]
    mask_value = np.transpose(np.tril(np.ones((query_window_sz, key_window_sz)) * np.NINF, -1), (1, 0))
    mask = tf.convert_to_tensor(value=mask_value, dtype=tf.float32)
    attention_mask = tf.tile(tf.reshape(mask, [-1, query_window_sz, key_window_sz]), [tf.shape(input=K)[0], 1, 1])

    kT = tf.transpose(keys, perm=[0, 2, 1])
    attention = tf.matmul(queries, kT)

    if use_mask:
        attention = tf.add(attention, attention_mask)
    attention = tf.nn.softmax(tf.divide(attention, np.sqrt(keys.get_shape()[2])))

    return attention
