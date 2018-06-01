import numpy as np
import tensorflow as tf
from guessing_sum_env import *

# TODO use the parameters of train_ddpg
HIDDEN_VECTOR_LEN = 9
NUM_AGENTS = 3
VECTOR_OBS_LEN = 5
OUTPUT_LEN = 10


class BiCNet:
    @staticmethod
    def base_build_network(observation):
        encoded = BiCNet.shared_dense_layer("encoder", observation, HIDDEN_VECTOR_LEN)

        hidden_agents = tf.unstack(encoded, NUM_AGENTS, 1)

        lstm_fw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0, name="lstm_fw_cell")
        lstm_bw_cell = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_VECTOR_LEN, forget_bias=1.0, name="lstm_bw_cell")
        outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, hidden_agents, dtype=tf.float32)
        outputs = tf.stack(outputs, 1)
        return outputs

    @staticmethod
    def actor_build_network(name, observation):
        with tf.variable_scope(name):
            outputs = BiCNet.base_build_network(observation)
            return BiCNet.shared_dense_layer("output_layer", observation, OUTPUT_LEN)


    @staticmethod
    def shared_dense_layer(name, observation, output_len):
        H = []
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for j in range(NUM_AGENTS):
                agent_obs = observation[:, j]
                agent_encoded = tf.layers.dense(agent_obs, output_len, name="dense")
                H.append(agent_encoded)
            H = tf.stack(H, 1)
        return H

    @staticmethod
    def critic_build_network(name, observation, action):
        with tf.variable_scope(name):
            outputs = BiCNet.base_build_network(observation)
            return BiCNet.shared_dense_layer("output_layer", observation, 1)


if __name__ == '__main__':
    tf.set_random_seed(42)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        BATCH_SIZE = 10

        observation = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="observation")
        actions = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN), name="actions")

        actor_out = BiCNet.actor_build_network("actor_network", observation)
        critic_out = BiCNet.critic_build_network("critic_network", observation, actions)

        sess.run(tf.global_variables_initializer())

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN), "== (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN)")

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, 1), "== (BATCH_SIZE, NUM_AGENTS, 1)")

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, OUTPUT_LEN), "== (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN)")

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((1, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, 1), "== (BATCH_SIZE, NUM_AGENTS, 1)")
