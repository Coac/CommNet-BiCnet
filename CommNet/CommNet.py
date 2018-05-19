import tensorflow as tf
from utils import *


class CommNet:
    def __init__(self):
        self.NUM_AGENTS = 5
        self.VECTOR_OBS_LEN = 10
        self.OUTPUT_LEN = 2

        H0 = tf.random_uniform(shape=(self.NUM_AGENTS, self.VECTOR_OBS_LEN), dtype="float32")
        H1 = self.comm_step("comm_step1", H0)
        H2 = self.comm_step("comm_step2", H1)
        self.out = self.output_layer(H2)

    def comm_step(self, name, H):
        with tf.variable_scope(name):
            w_H = tf.get_variable(name='w_H', shape=self.VECTOR_OBS_LEN, initializer=tf.constant_initializer(10))
            w_C = tf.get_variable(name='w_C', shape=self.VECTOR_OBS_LEN, initializer=tf.constant_initializer(1))

            normalized_w_C = tf.divide(w_C, self.NUM_AGENTS - 1)

            w_C_matrix = tf.reshape(tf.tile(normalized_w_C, [self.NUM_AGENTS]),
                                    (1, self.NUM_AGENTS * self.VECTOR_OBS_LEN))
            w_C_matrix = tf.tile(w_C_matrix, [self.NUM_AGENTS, 1])

            w_H_minus_w_C_matrix = block_diagonal(
                [tf.reshape(tf.subtract(w_H, normalized_w_C), (1, self.VECTOR_OBS_LEN)) for j in
                 range(self.NUM_AGENTS)])
            T = tf.add(w_C_matrix, w_H_minus_w_C_matrix)

            H_repeated = tf.tile(H, [1, self.NUM_AGENTS])

            return tf.tanh(tf.reshape(tf.reduce_sum(H_repeated * T, axis=0), H.shape))

    def output_layer(self, H):
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            w = tf.get_variable(name='w', shape=(self.VECTOR_OBS_LEN, self.OUTPUT_LEN),
                                initializer=tf.constant_initializer(10))

            outputs = []
            for j in range(self.NUM_AGENTS):
                h = tf.slice(H, [j, 0], [1, self.VECTOR_OBS_LEN])

                outputs.append(tf.tanh(tf.matmul(h, w))[0])
            return outputs


if __name__ == '__main__':
    tf.set_random_seed(42)

    commNet = CommNet()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter("summaries", sess.graph)

        sess.run(tf.global_variables_initializer())

        res = sess.run(commNet.out)
        for i in res:
            print('-----')
            print(i)

        writer.close()
