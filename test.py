import tensorflow as tf
from tensorflow.contrib import layers

def matrix_batch_vectors_mul(mat, batch_vectors, shape_after_mul):
    '''

    :param mat: N*N
    :param batch_vectors: batch_size*sen_num*N 
    :param shape_after_mul: batch_size*sen_num*N
    :return: new tensor shape is batch_size*sen_num*N
    '''
    vectors = tf.reshape(batch_vectors, [-1, batch_vectors.shape[-1].value])
    res = tf.matmul(mat, vectors, transpose_a=False, transpose_b=True)
    return tf.reshape(tf.transpose(res), shape_after_mul)

def batch_vectors_vector_mul(batch_vectors, vector, shape_after_mul):
    expand_vec = tf.expand_dims(vector, -1)
    mat_vec = tf.reshape(batch_vectors, [-1, batch_vectors.get_shape()[-1].value])
    res = tf.matmul(mat_vec, expand_vec)
    return tf.reshape(res, shape_after_mul)

def _word_attention(encoded_words):
    with tf.variable_scope('word_attention'):
        encoded_word_dims = 2

        word_context = tf.Variable(tf.truncated_normal([2]), name='word_context')
        W_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims, encoded_word_dims]), name='word_context_weights')
        b_word = tf.Variable(tf.truncated_normal(shape=[encoded_word_dims]), name='word_context_bias')
        U_w = tf.tanh(matrix_batch_vectors_mul(W_word, encoded_words, [-1, 4, encoded_word_dims]) + b_word)
        word_logits = batch_vectors_vector_mul(U_w, word_context, [-1, 4])
        word_logit = tf.reduce_sum(tf.multiply(U_w, word_context), axis=2, keep_dims=True)
        return word_logits ,word_logit

x3 = tf.constant([[[[1, 2], [2, 2], [3, 2], [4, 2]],
                   [[5, 2], [6, 2], [7, 2], [8, 2]],
                   [[9, 2], [10, 2], [11, 2], [12 ,2]]],

                  [[[1, 2], [2, 2], [3, 2], [4 ,2]],
                   [[5 ,2], [6, 2], [7, 2], [8 ,2]],
                   [[9, 2], [10, 2], [11 ,2], [12, 2]]]], tf.float32)

x3 = tf.reshape(x3, [-1, 4 ,2])

W_word = tf.constant([[1, 2], [3, 4]], tf.float32)
b_word = tf.constant([[0.1], [0.2]])
u_context = tf.constant(value=2, shape=[2], dtype=tf.float32)
#u_w = tf.matmul(x3, W_word) + b_word
u_ww = matrix_batch_vectors_mul(W_word, x3, [-1, 4, 2])
uu = tf.nn.softmax(batch_vectors_vector_mul(u_ww, u_context, [-1, 4]))
uu1 = tf.nn.softmax(tf.reduce_sum(tf.multiply(u_ww, u_context), axis=2, keep_dims=False), dim=1)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    xx, xy, xz, xw, xu= sess.run([x3,  u_ww, uu, uu1, u_context])
    print xx, xy, xz, xw, xu
