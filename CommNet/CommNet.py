import numpy as np
import tensorflow as tf
from utils import *


class CommNet:
    def __init__(self, sess, NUM_AGENTS, VECTOR_OBS_LEN, OUTPUT_LEN, learning_rate=0.01):
        self.NUM_AGENTS = NUM_AGENTS
        self.VECTOR_OBS_LEN = VECTOR_OBS_LEN
        self.OUTPUT_LEN = OUTPUT_LEN

        self.observation = tf.placeholder(tf.float32, (self.NUM_AGENTS, self.VECTOR_OBS_LEN), name="observation")
        H1 = self.comm_step("comm_step1", self.observation)
        H2 = self.comm_step("comm_step2", H1)
        self.out = self.output_layer(H2)

        self.sess = sess or tf.get_default_session()

        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []

        self.target = tf.placeholder(1, name="target")
        self.loss = -self.dist.log_prob(self.actions) * self.target
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

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
            w = tf.get_variable(name='w_out', shape=(self.VECTOR_OBS_LEN, self.OUTPUT_LEN),
                                initializer=tf.constant_initializer(10))

            b = tf.get_variable(name='b_out', shape=self.OUTPUT_LEN, initializer=tf.constant_initializer(100))

            actions = []
            for j in range(self.NUM_AGENTS):
                h = tf.slice(H, [j, 0], [1, self.VECTOR_OBS_LEN])

                means = tf.matmul(h, w) + b
                stds = [1.0 for i in range(self.OUTPUT_LEN)]

                self.dist = tf.distributions.Normal(means, stds)

                action = tf.squeeze(self.dist.sample(1))
                actions.append(action)

            self.actions = tf.stack(actions)
            return self.actions

    def predict(self, observation):
        return np.array(self.sess.run(self.out, feed_dict={self.observation: observation}))

    def store_transition(self, s, a, r):
        """
        s -- state
        a -- actions
        r -- scalar reward
        """

        self.ep_observations.append(s)
        self.ep_actions.append(a)
        self.ep_rewards.append(r)

    def train_step(self):
        for i in range(len(self.ep_observations)):
            state = self.ep_observations[i]
            actions = self.ep_actions[i]
            reward = self.ep_rewards[i]

            feed_dict = {self.observation: state, self.target: reward, self.actions: actions}
            _, loss = sess.run([self.train_op, self.loss], feed_dict)

        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []


if __name__ == '__main__':
    tf.set_random_seed(42)

    NUM_AGENTS = 5
    VECTOR_OBS_LEN = 10
    OUTPUT_LEN = 2

    with tf.Session() as sess:
        commNet = CommNet(sess, NUM_AGENTS, VECTOR_OBS_LEN, OUTPUT_LEN)

        writer = tf.summary.FileWriter("summaries", sess.graph)

        sess.run(tf.global_variables_initializer())

        outputs = commNet.predict(np.random.random((NUM_AGENTS, VECTOR_OBS_LEN)))
        print(outputs.shape)
        print(outputs)

        writer.close()
