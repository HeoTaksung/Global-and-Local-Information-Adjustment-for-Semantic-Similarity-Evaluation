import tensorflow as tf
from keras.layers import *
from keras import backend as K
from keras.models import Model
from keras_self_attention import SeqSelfAttention


def local_manhattan_distance(left, right):
    return tf.reduce_mean(K.exp(-K.sum(K.abs(left-right), axis=2, keepdims=True)), axis=1)


def global_manhattan_distance(left, right):
    return K.exp(-K.sum(K.abs(left-right), axis=1, keepdims=True))


def squash(vectors, axis=-1):

    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())

    return scale * vectors


def ratio(v1, v2, a, b):
    return a*v1 + b*v2


def global_local_share_model(self, summary=True):
    left_input = Input(shape=(self.max_len,))
    right_input = Input(shape=(self.max_len,))

    if self.exist_emb:
        embedding = Embedding(self.vocab_size, self.embedding_size, weights=[self.pretrain_vec], trainable=False)
    else:
        embedding = Embedding(self.vocab_size, self.embedding_size)

    left_emd = embedding(left_input)
    right_emd = embedding(right_input)

    lstm = Bidirectional(LSTM(self.lstm_units, return_sequences=True, return_state=True))

    left_lstm, l_fh, l_fc, l_bh, l_bc = lstm(left_emd)
    right_lstm, r_fh, r_fc, r_bh, r_bc = lstm(right_emd)

    left_attention = SeqSelfAttention(attention_activation='tanh')(left_lstm)
    right_attention = SeqSelfAttention(attention_activation='tanh')(right_lstm)

    left_attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(left_attention)
    right_attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(right_attention)

    cnn = Conv2D(filters=self.cnn_filters, kernel_size=(self.kernel_size, 2*self.lstm_units),
                 strides=1, padding='valid', activation='relu')

    left_cnn = cnn(left_attention)
    right_cnn = cnn(right_attention)

    dim_capsule = 8
    len_ui = 32

    primary_caps = Conv2D(filters=dim_capsule * len_ui, kernel_size=(K.int_shape(left_cnn)[1], 1),
                          strides=2, padding='valid')

    left_primary = primary_caps(left_cnn)
    right_primary = primary_caps(right_cnn)

    left_primary = Reshape(target_shape=[dim_capsule, len_ui])(left_primary)
    right_primary = Reshape(target_shape=[dim_capsule, len_ui])(right_primary)

    left_primary = Lambda(squash, name='left_primary_caps_squash')(left_primary)
    right_primary = Lambda(squash, name='right_primary_caps_squash')(right_primary)

    l_weight = concatenate([l_fh, l_bh], axis=-1)
    r_weight = concatenate([r_fh, r_bh], axis=-1)

    distance1 = Lambda(function=lambda x: local_manhattan_distance(x[0], x[1]),
                       output_shape=lambda x: (x[0][0], 1))([left_primary, right_primary])
    distance2 = Lambda(function=lambda x: global_manhattan_distance(x[0], x[1]),
                       output_shape=lambda x: (x[0][0], 1))([l_weight, r_weight])

    distance = Lambda(function=lambda x: ratio(x[0], x[1], self.local_ratio, self.global_ratio),
                      output_shape=lambda x: (x[0][0], 1), name='mean_distance')([distance1, distance2])

    model = Model(inputs=[left_input, right_input], outputs=[distance])

    if summary:
        model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])

    return model


def global_model(self, summary=True):
    left_input = Input(shape=(self.max_len,))
    right_input = Input(shape=(self.max_len,))

    if self.exist_emb:
        embedding = Embedding(self.vocab_size, self.embedding_size, weights=[self.pretrain_vec], trainable=False)
    else:
        embedding = Embedding(self.vocab_size, self.embedding_size)

    left_emd = embedding(left_input)
    right_emd = embedding(right_input)

    lstm = Bidirectional(LSTM(128))

    left_lstm = lstm(left_emd)
    right_lstm = lstm(right_emd)

    distance = Lambda(function=lambda x: global_manhattan_distance(x[0], x[1]),
                      output_shape=lambda x: (x[0][0], 1))([left_lstm, right_lstm])

    model = Model(inputs=[left_input, right_input], outputs=[distance])

    if summary:
        model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])

    return model


def local_model(self, summary=True):
    left_input = Input(shape=(self.max_len,))
    right_input = Input(shape=(self.max_len,))

    if self.exist_emb:
        embedding = Embedding(self.vocab_size, self.embedding_size, weights=[self.pretrain_vec], trainable=False)
    else:
        embedding = Embedding(self.vocab_size, self.embedding_size)

    left_emd = embedding(left_input)
    right_emd = embedding(right_input)

    lstm = Bidirectional(LSTM(self.lstm_units, return_sequences=True, return_state=True))

    left_lstm, l_fh, l_fc, l_bh, l_bc = lstm(left_emd)
    right_lstm, r_fh, r_fc, r_bh, r_bc = lstm(right_emd)

    left_attention = SeqSelfAttention(attention_activation='tanh')(left_lstm)
    right_attention = SeqSelfAttention(attention_activation='tanh')(right_lstm)

    left_attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(left_attention)
    right_attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(right_attention)

    cnn = Conv2D(filters=self.cnn_filters, kernel_size=(3, 2 * self.lstm_units),
                 strides=1, padding='valid', activation='relu')

    left_cnn = cnn(left_attention)
    right_cnn = cnn(right_attention)

    dim_capsule = 8
    len_ui = 32

    primary_caps = Conv2D(filters=dim_capsule * len_ui, kernel_size=(K.int_shape(left_cnn)[1], 1),
                          strides=2, padding='valid')

    left_primary = primary_caps(left_cnn)
    right_primary = primary_caps(right_cnn)

    left_primary = Reshape(target_shape=[dim_capsule, len_ui])(left_primary)
    right_primary = Reshape(target_shape=[dim_capsule, len_ui])(right_primary)

    left_primary = Lambda(squash, name='left_primary_caps_squash')(left_primary)
    right_primary = Lambda(squash, name='right_primary_caps_squash')(right_primary)

    distance = Lambda(function=lambda x: local_manhattan_distance(x[0], x[1]),
                      output_shape=lambda x: (x[0][0], 1))([left_primary, right_primary])

    model = Model(inputs=[left_input, right_input], outputs=[distance])

    if summary:
        model.summary()

    model.compile(loss="mean_squared_error", optimizer="adam", metrics=['accuracy'])

    return model
