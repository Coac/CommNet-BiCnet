import numpy as np
import tensorflow as tf
from guessing_sum_env import *

# TODO use the parameters of train_ddpg
HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 2
VECTOR_OBS_LEN = 1
OUTPUT_LEN = 1


class CommNet:
    @staticmethod
    def base_build_network(observation):
        # H0 = CommNet.encoder(observation)
        H0 = observation
        C0 = tf.zeros(tf.shape(H0), name="C0")
        H1, C1 = CommNet.comm_step("comm_step1", H0, C0)
        H2, _ = CommNet.comm_step("comm_step2", H1, C1, H0)
        # H3, _ = CommNet.comm_step("comm_step3", H2, C2, H0)
        return H2

    @staticmethod
    def actor_build_network(name, observation):
        with tf.variable_scope(name):
            H = CommNet.base_build_network(observation)
            return CommNet.actor_output_layer(H)

    @staticmethod
    def critic_build_network(name, observation, action):
        with tf.variable_scope(name):
            H = CommNet.base_build_network(observation)
            return CommNet.critic_output_layer(H, action)

    @staticmethod
    def encoder(s):
        H = []
        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for j in range(NUM_AGENTS):
                encoded = tf.layers.dense(tf.reshape(s[j], (1, VECTOR_OBS_LEN)), HIDDEN_VECTOR_LEN, name="dense")
                H.append(tf.squeeze(encoded))
            H = tf.stack(H)
            H = tf.reshape(H, (NUM_AGENTS, HIDDEN_VECTOR_LEN))

        return H

    @staticmethod
    def module(h, c):
        with tf.variable_scope("module", reuse=tf.AUTO_REUSE):
            w_H = tf.get_variable(name='w_H', shape=HIDDEN_VECTOR_LEN,
                                  initializer=tf.contrib.layers.xavier_initializer())
            w_C = tf.get_variable(name='w_C', shape=HIDDEN_VECTOR_LEN,
                                  initializer=tf.contrib.layers.xavier_initializer())

            tf.summary.histogram('w_H', w_H)
            tf.summary.histogram('w_C', w_C)

            return tf.tanh(tf.multiply(w_H, h) + tf.multiply(w_C, c))

    @staticmethod
    def comm_step(name, H, C, H0_skip_con=None):
        batch_size = tf.shape(H)[0]
        with tf.variable_scope(name):
            next_H = tf.zeros(shape=(batch_size, 0, HIDDEN_VECTOR_LEN))
            for j in range(NUM_AGENTS):
                h = H[:, j]
                c = C[:, j]

                next_h = CommNet.module(h, c)  # shape (BATCH_SIZE, HIDDEN_VECTOR_LEN)
                next_H = tf.concat([next_H, tf.reshape(next_h, (batch_size, 1, HIDDEN_VECTOR_LEN))], 1)

            next_H = tf.identity(next_H, "H")

            if H0_skip_con is not None:
                next_H = tf.add(next_H, H0_skip_con)

            if NUM_AGENTS > 1:
                next_C = tf.zeros(shape=(batch_size, 0, HIDDEN_VECTOR_LEN))
                for j1 in range(NUM_AGENTS):
                    next_c = []
                    for j2 in range(NUM_AGENTS):
                        if j1 != j2:
                            next_c.append(next_H[:, j2])
                    next_c = tf.reduce_mean(tf.stack(next_c), 0)
                    next_C = tf.concat([next_C, tf.reshape(next_c, (batch_size, 1, HIDDEN_VECTOR_LEN))], 1)
            else:
                next_C = C

            return next_H, tf.identity(next_C, "C")

    @staticmethod
    def actor_output_layer(H):
        with tf.variable_scope("actor_output"):
            w_out = tf.get_variable(name='w_out', shape=(HIDDEN_VECTOR_LEN, OUTPUT_LEN),
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.get_variable(name='b_out', shape=OUTPUT_LEN, initializer=tf.zeros_initializer())

            tf.summary.histogram('w_out', w_out)
            tf.summary.histogram('b_out', b_out)

            batch_size = tf.shape(H)[0]

            actions = []
            for j in range(NUM_AGENTS):
                h = tf.slice(H, [0, j, 0], [batch_size, 1, HIDDEN_VECTOR_LEN])
                w_out_batch = tf.tile(tf.expand_dims(w_out, axis=0), [batch_size, 1, 1])
                action =  tf.squeeze(tf.matmul(h, w_out_batch) + b_out, [1])

                actions.append(action)
            actions = tf.stack(actions, name="actions", axis=1)

        return actions

    @staticmethod
    def critic_output_layer(H, action):
        with tf.variable_scope("critic_output", reuse=tf.AUTO_REUSE):
            baseline = tf.layers.dense(inputs=tf.concat([H, action], 2),
                                       units=1,
                                       activation=tf.tanh,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            baseline = tf.squeeze(baseline, [2])
            baseline = tf.layers.dense(inputs=baseline,
                                       units=1,
                                       kernel_initializer=tf.contrib.layers.xavier_initializer())
            tf.summary.histogram("w_baseline", tf.get_variable("dense/kernel"))

            return baseline


if __name__ == '__main__':
    tf.set_random_seed(42)

    tf.reset_default_graph()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        BATCH_SIZE = 10

        observation = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN))
        actions = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN))

        actor_out = CommNet.actor_build_network("actor_network", observation)
        critic_out = CommNet.critic_build_network("critic_network", observation, actions)

        sess.run(tf.global_variables_initializer())

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))

        feed_dict = {observation: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((BATCH_SIZE, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (BATCH_SIZE, 1))

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN))}
        print(sess.run(actor_out, feed_dict=feed_dict).shape, "==", (1, NUM_AGENTS, OUTPUT_LEN))

        feed_dict = {observation: np.random.random_sample((1, NUM_AGENTS, VECTOR_OBS_LEN)),
                     actions: np.random.random_sample((1, NUM_AGENTS, OUTPUT_LEN))}
        print(sess.run(critic_out, feed_dict=feed_dict).shape, "==", (1, 1))
