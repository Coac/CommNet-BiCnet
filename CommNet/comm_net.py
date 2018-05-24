from datetime import datetime

import numpy as np
import tensorflow as tf
from guessing_sum_env import *

HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 1
VECTOR_OBS_LEN = 1
OUTPUT_LEN = 1


class CommNet:
    @staticmethod
    def base_build_network(observation):
        # H0 = CommNet.encoder(observation)
        H0 = tf.reshape(observation, (NUM_AGENTS, HIDDEN_VECTOR_LEN))
        C0 = tf.zeros((NUM_AGENTS, HIDDEN_VECTOR_LEN), name="C0")
        H1, C1 = CommNet.comm_step("comm_step1", H0, C0)
        H2, _ = CommNet.comm_step("comm_step2", H1, C1, H0)
        # H3, _ = CommNet.comm_step("comm_step3", H2, C2, H0)
        return H2

    @staticmethod
    def actor_build_network(name, observation):
        with tf.variable_scope(name):
            H = CommNet.base_build_network(observation)
            return H
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

            # w_H = tf.Print(w_H, [w_H], message=tf.get_default_graph().get_name_scope() + "w_H")

            tf.summary.histogram('w_H', w_H)
            tf.summary.histogram('w_C', w_C)

            return tf.tanh(tf.multiply(w_H, h) + tf.multiply(w_C, c))

    @staticmethod
    def comm_step(name, H, C, H0_skip_con=None):
        with tf.variable_scope(name):
            next_H = []
            for j in range(NUM_AGENTS):
                h = H[j]
                c = C[j]

                next_h = CommNet.module(h, c)
                next_H.append(next_h)

            next_H = tf.stack(next_H)
            next_H = tf.identity(next_H, "H")

            if H0_skip_con is not None:
                next_H = tf.add(next_H, H0_skip_con)

            next_C = []
            for j1 in range(NUM_AGENTS):
                next_c = []
                for j2 in range(NUM_AGENTS):
                    if j1 != j2:
                        next_c.append(next_H[j2])
                next_c = tf.reduce_mean(tf.stack(next_c), 0)
                next_c = tf.where(tf.is_nan(next_c), tf.zeros_like(next_c), next_c)

                next_C.append(next_c)

            return next_H, tf.identity(next_C, "C")

    @staticmethod
    def actor_output_layer(H):
        with tf.variable_scope("actor_output"):
            w_out = tf.get_variable(name='w_out', shape=(HIDDEN_VECTOR_LEN, OUTPUT_LEN),
                                    initializer=tf.contrib.layers.xavier_initializer())
            b_out = tf.get_variable(name='b_out', shape=OUTPUT_LEN, initializer=tf.zeros_initializer())

            tf.summary.histogram('w_out', w_out)
            tf.summary.histogram('b_out', b_out)

            actions = []
            for j in range(NUM_AGENTS):
                h = tf.slice(H, [j, 0], [1, HIDDEN_VECTOR_LEN])
                action = tf.squeeze(tf.matmul(h, w_out) + b_out)
                actions.append(action)
            actions = tf.stack(actions, name="actions")

        return actions

    @staticmethod
    def critic_output_layer(H, action):
        with tf.variable_scope("critic_output", reuse=tf.AUTO_REUSE):
            # TODO merge action
            # baseline = tf.layers.dense(tf.reshape(H, (1, NUM_AGENTS * HIDDEN_VECTOR_LEN)), units=1,
            #                                    kernel_initializer=tf.contrib.layers.xavier_initializer())

            baseline = tf.layers.dense(
                tf.reshape([H, action], (1, NUM_AGENTS * HIDDEN_VECTOR_LEN + NUM_AGENTS * OUTPUT_LEN)), units=1,
                kernel_initializer=tf.contrib.layers.xavier_initializer())

            baseline = tf.squeeze(baseline)

            tf.summary.histogram("w_baseline", tf.get_variable("dense/kernel"))

            return baseline


if __name__ == '__main__':
    tf.set_random_seed(42)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        commNet = CommNet(sess, NUM_AGENTS, VECTOR_OBS_LEN, OUTPUT_LEN)

        writer = tf.summary.FileWriter("summaries/" + datetime.now().strftime('%d-%m-%y %H%M'), sess.graph)
        sess.run(tf.global_variables_initializer())

        env = GuessingSumEnv(NUM_AGENTS)
        env.seed(0)

        for episode in range(1000000):
            observations = env.reset()
            while True:
                actions = commNet.predict(observations)
                _, rewards, done, _ = env.step(np.reshape(actions, (NUM_AGENTS, 1)))
                reward = -(rewards ** 2).mean()
                commNet.store_transition(observations, actions, reward)

                if episode % 1000 == 0:
                    feed_dict = {"observation:0": observations, "reward:0": reward, "output/actions:0": actions}

                    merged_summary = tf.summary.merge_all()
                    summary = sess.run(merged_summary, feed_dict=feed_dict)
                    writer.add_summary(summary, episode)
                    print("reward: ", reward, " | actions: ", actions, " | expected_output:", np.sum(observations))

                if done:
                    commNet.train_step()
                    break

        writer.close()
