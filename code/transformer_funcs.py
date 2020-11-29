import numpy as np
import tensorflow as tf


class Transformer_Block(tf.keras.layers.Layer):
    def __init__(self, emb_size, is_decoder, multi_headed=False):
        super(Transformer_Block, self).__init__()

        # Self Attention Layers
        if not multi_headed:
            self.self_attention = Attention_Head(emb_size, emb_size, use_mask=is_decoder)
            if self.is_decoder:
                # Encoder-Decoder Attention for Decoders
                self.context_attention = Attention_Head(emb_size, emb_size, use_mask=False)
        else:
            self.self_attention = Multi_Headed(emb_size, use_mask=is_decoder)
            if self.is_decoder:
                # Encoder-Decoder Attention for Decoders
                self.context_attention = Multi_Headed(emb_size, use_mask=False)

        # Normalization Layer
        self.layer_norm = tf.keras.layers.LayerNormalization(axis=-1)
        # Feed Forward Layers
        self.dense1 = tf.keras.layers.Dense(emb_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(emb_size)

    @tf.function
    def call(self, inputs, context=None):
        """
        This function calls a forward pass on a transformer block

        :param inputs: input sequence tensor (batch size, window size, embedding size)
        :param context:
        :return: a tensor of predictions for each time step (batch size, window size, embedding size)
        """

        # Self Attention
        attention_out = self.self_attention(inputs, inputs, inputs)
        attention_out += inputs
        # Normalizing
        attention_norm = self.layer_norm(attention_out)

        if self.is_decoder:
            assert context is not None
            # Encoder-Decoder Attention
            context_attention_out = self.context_attention(context, context, attention_norm)
            context_attention_out += attention_norm
            attention_norm = self.layer_norm(context_attention_out)

        # Feed Forward
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

        # Whether or not to use a mask.  Set to true when decoding
        self.use_mask = use_mask

        # The trainable weight matrices for keys, values, and queries
        # Multiplying the word embeddings by each of these will produce the key, value, and query vectors
        self.queries = self.add_weight(name='Q', shape=(input_size, output_size), dtype=tf.float32, trainable=True)
        self.keys = self.add_weight(name='K', shape=(input_size, output_size), dtype=tf.float32, trainable=True)
        self.values = self.add_weight(name='V', shape=(input_size, output_size), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, key_inputs, value_inputs, query_inputs):
        """
        This function conducts the self attention step of a Transformer Block, using single-headed attentionf

        :param key_inputs: the keys
        :param value_inputs: they values
        :param query_inputs: the queries
        :return: new word embeddings that are a result of self-attention.  These should be passed to a FF Layer
        """
        # multiply the word embeddings by the weight matrices to produce the key, value, and query vectors
        Q = tf.tensordot(query_inputs, self.queries, ((2), (0)))
        K = tf.tensordot(key_inputs, self.keys, ((2), (0)))
        V = tf.tensordot(value_inputs, self.values, ((2), (0)))

        # Compute the matrix of each words attention to each other word
        attention_matrix = Attention_Matrix(K, Q, use_mask=self.user_mask)
        # Matmul this this the value vector
        Z = tf.matmul(attention_matrix, V)

        return Z


class Multi_Headed(tf.keras.layers.Layer):
    def __init__(self, size, use_mask):
        super(Multi_Headed, self).__init__()

        self.use_mask = use_mask # True for the Decoders

        self.vector_size = 20
        self.num_heads = 3
        # The trainable weight matrices for keys, values, and queries
        # Multiplying the word embeddings by each of these will produce the key, value, and query vectors
        self.queries1 = self.add_weight(name='Q1', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.keys1 = self.add_weight(name='K1', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.values1 = self.add_weight(name='V1', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)

        self.queries2 = self.add_weight(name='Q2', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.keys2 = self.add_weight(name='K2', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.values2 = self.add_weight(name='V2', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)

        self.queries3 = self.add_weight(name='Q3', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.keys3 = self.add_weight(name='K3', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)
        self.values3 = self.add_weight(name='V3', shape=(size, self.vector_size), dtype=tf.float32, trainable=True)

        self.WO = self.add_weight(name='WO', shape=(self.vector_size*self.num_heads), dtype=tf.float32, trainable=True)

    @tf.function
    def call(self, key_inputs, value_inputs, query_inputs):
        """
        This function computes the result of multi-headed attenion.  In this case we are using 3 heads

        :param key_inputs:
        :param value_inputs:
        :param query_inputs:
        :return:
        """

        Q1 = tf.tensordot(query_inputs, self.queries1, ((2), (0)))
        K1 = tf.tensordot(key_inputs, self.keys1, ((2), (0)))
        V1 = tf.tensordot(value_inputs, self.values1, ((2), (0)))

        Q2 = tf.tensordot(query_inputs, self.queries2, ((2), (0)))
        K2 = tf.tensordot(key_inputs, self.keys2, ((2), (0)))
        V2 = tf.tensordot(value_inputs, self.values2, ((2), (0)))

        Q3 = tf.tensordot(query_inputs, self.queries3, ((2), (0)))
        K3 = tf.tensordot(key_inputs, self.keys3, ((2), (0)))
        V3 = tf.tensordot(value_inputs, self.values3, ((2), (0)))

        matrix1 = Attention_Matrix(K1, Q1, self.use_mask)
        matrix2 = Attention_Matrix(K2, Q2, self.use_mask)
        matrix3 = Attention_Matrix(K3, Q3, self.use_mask)

        Z1 = tf.matmul(matrix1, V1)
        Z2 = tf.matmul(matrix2, V2)
        Z3 = tf.matmul(matrix3, V3)

        Z = tf.concat([Z1, Z2, Z3], axis=-1)
        Z = tf.matmul(Z, self.WO)

        return Z


def Attention_Matrix(keys, queries, use_mask=False):
    """
    This function computes the attention matrix, which is a matrix of dim (num queries, num keys)
    the (i, j) entry corresponds to the score of the jth value vector when computing the attention for the ith word
    This matrix will be matmuled with the value matrix to compute the z vecotr for each word in the sequence

    :param keys: a matrix of keys
    :param queries: a matrix of queries
    :param use_mask: a bool, whether to mask the attention matrix of not (used in the decoder so it can't peak ahead)
    :return: a matrix of scores
    """
    # Get the window size for the query and key vectors
    query_window_sz = queries.get_shape()[1]
    key_window_sz = keys.get_shape()[1]

    # Compute the (similarity) score for each query.  There are many functions that we can utilize, and
    # We use a scaled dot product
    kT = tf.transpose(keys, perm=[0, 2, 1])
    attention = tf.matmul(queries, kT)
    attention = tf.divide(attention, np.sqrt(keys.get_shape()[2]))

    # For the decoder
    if use_mask:
        # Compute the mask vector:
        # Create a matrix of negative infinity, then zero the elements above and including the main diagonal
        # Then transpose across 1st and 2nd dimension
        mask_value = np.transpose(np.tril(np.ones((query_window_sz, key_window_sz)) * np.NINF, -1), (1, 0))
        # Convert this np array to a tensor
        mask = tf.convert_to_tensor(value=mask_value, dtype=tf.float32)
        # reshape the matrix to (batch size, query window size, key window size)
        # Then tile it across the first dimension number of keys times
        attention_mask = tf.tile(tf.reshape(mask, [-1, query_window_sz, key_window_sz]),
                                 [tf.shape(input=keys)[0], 1, 1])
        # Mask the attention matrix
        attention = tf.add(attention, attention_mask)

    # Divide the score to stabilize gradients, then softmax to normalize the scores
    attention = tf.nn.softmax(attention)

    return attention
