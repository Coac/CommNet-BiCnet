import numpy as np
import tensorflow as tf
from guessing_sum_env import *

# TODO use the parameters of train_ddpg
HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 20
VECTOR_OBS_LEN = 1
OUTPUT_LEN = 1


class BiCNet:
    @staticmethod
    def base_build_network(observation):
        H0 = observation
        return H0

    @staticmethod
    def actor_build_network(name, observation):
        x = tf.unstack(observation, NUM_AGENTS, 1)
        with tf.variable_scope(name):
            num_hidden = 10
            lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
            lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
            outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
            print(tf.stack(outputs, 1))

            # TODO concatenate fw + bw to form output
            return tf.stack(outputs, 1)

    @staticmethod
    def critic_build_network(name, observation, action):
        with tf.variable_scope(name):
            return tf.constant(1)


if __name__ == '__main__':
    tf.set_random_seed(42)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        BATCH_SIZE = 10

        observation = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN))
        actions = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN))

        actor_out = BiCNet.actor_build_network("actor_network", observation)
        critic_out = BiCNet.critic_build_network("critic_network", observation, actions)

        sess.run(tf.global_variables_initializer())

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, 1))

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, OUTPUT_LEN))

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((1, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (1, 1))
