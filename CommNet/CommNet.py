from datetime import datetime

import numpy as np
import tensorflow as tf
from guessing_sum_env import *
from utils import *

HIDDEN_VECTOR_LEN = 1


class CommNet:
    def __init__(self, sess, NUM_AGENTS, VECTOR_OBS_LEN, OUTPUT_LEN, learning_rate=0.0001):
        self.NUM_AGENTS = NUM_AGENTS
        self.VECTOR_OBS_LEN = VECTOR_OBS_LEN
        self.OUTPUT_LEN = OUTPUT_LEN

        self.observation = tf.placeholder(tf.float32, (self.NUM_AGENTS, self.VECTOR_OBS_LEN), name="observation")

        H0 = self.encoder(self.observation)
        C0 = tf.zeros((self.NUM_AGENTS, HIDDEN_VECTOR_LEN), name="C0")
        H1, C1 = self.comm_step("comm_step1", H0, C0)
        H2, C2 = self.comm_step("comm_step2", H1, C1)
        H3, _ = self.comm_step("comm_step3", H2, C2, H0)

        self.out = self.output_layer(H3)

        self.sess = sess or tf.get_default_session()

        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []

        self.reward = tf.placeholder(tf.float32, shape=(), name="reward")

        with tf.name_scope("loss"):
            alpha = 1
            self.policy_loss = 0
            actions = tf.reshape(self.actions, (self.NUM_AGENTS, self.OUTPUT_LEN))
            for j in range(self.NUM_AGENTS):
                normal_dist = self.normal_dists[j]
                log_prob = normal_dist.log_prob(actions[j])
                self.policy_loss -= tf.squeeze(tf.reduce_mean(log_prob)) * (self.reward - self.baseline)

            self.baseline_loss = alpha * tf.square(self.reward - self.baseline)
            self.total_loss = self.policy_loss + self.baseline_loss

            tf.summary.scalar('baseline_loss', self.baseline_loss)
            tf.summary.scalar('policy_loss', self.policy_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            tf.summary.scalar('reward', self.reward)
            tf.summary.scalar('baseline', self.baseline)

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            gradients = self.optimizer.compute_gradients(self.total_loss)
            clipped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
            self.train_op = self.optimizer.apply_gradients(clipped_gradients)

            # self.train_op = tf.gradients(self.total_loss, tf.trainable_variables()[:len(tf.trainable_variables()) -2])

    def encoder(self, s):
        H = []

        with tf.variable_scope("encoder", reuse=tf.AUTO_REUSE):
            for j in range(self.NUM_AGENTS):
                encoded = tf.layers.dense(tf.reshape(s[j], (1, self.VECTOR_OBS_LEN)), HIDDEN_VECTOR_LEN, name="dense")
                H.append(tf.squeeze(encoded))
            H = tf.stack(H)
            H = tf.reshape(H, (self.NUM_AGENTS, HIDDEN_VECTOR_LEN))

        return H

    def module(self, h, c):
        with tf.variable_scope("module", reuse=tf.AUTO_REUSE):
            w_H = tf.get_variable(name='w_H', shape=HIDDEN_VECTOR_LEN,
                                  initializer=tf.contrib.layers.xavier_initializer())
            w_C = tf.get_variable(name='w_C', shape=HIDDEN_VECTOR_LEN,
                                  initializer=tf.contrib.layers.xavier_initializer())

            tf.summary.histogram('w_H', w_H)
            tf.summary.histogram('w_C', w_C)

            return tf.tanh(tf.multiply(w_H, h) + tf.multiply(w_C, c))

    def comm_step(self, name, H, C, H0_skip_con=None):
        with tf.variable_scope(name):
            next_H = []
            for j in range(self.NUM_AGENTS):
                h = H[j]
                c = C[j]
                next_h = self.module(h, c)
                next_H.append(next_h)

            next_H = tf.stack(next_H)
            next_H = tf.identity(next_H, "H")

            if H0_skip_con is not None:
                next_H = tf.add(next_H, H0_skip_con)

            next_C = []
            for j1 in range(self.NUM_AGENTS):
                next_c = []
                for j2 in range(self.NUM_AGENTS):
                    if j1 != j2:
                        next_c.append(next_H[j2])
                next_c = tf.reduce_mean(tf.stack(next_c), 0)
                next_C.append(next_c)

            return next_H, tf.identity(next_C, "C")

    def output_layer(self, H):
        with tf.variable_scope("output", reuse=tf.AUTO_REUSE):
            w_means = tf.get_variable(name='w_means_out', shape=(HIDDEN_VECTOR_LEN, self.OUTPUT_LEN),
                                      initializer=tf.contrib.layers.xavier_initializer())
            b_means = tf.get_variable(name='b_means_out', shape=self.OUTPUT_LEN, initializer=tf.zeros_initializer())

            w_stds = tf.get_variable(name='w_stds_out', shape=(HIDDEN_VECTOR_LEN, self.OUTPUT_LEN),
                                     initializer=tf.contrib.layers.xavier_initializer())
            b_stds = tf.get_variable(name='b_stds_out', shape=self.OUTPUT_LEN)

            tf.summary.histogram('w_means_out', w_means)
            tf.summary.histogram('b_means_out', w_stds)
            tf.summary.histogram('w_stds', w_stds)
            tf.summary.histogram('b_stds', b_stds)

            actions = []
            self.normal_dists = []
            for j in range(self.NUM_AGENTS):
                h = tf.slice(H, [j, 0], [1, HIDDEN_VECTOR_LEN])

                means = tf.matmul(h, w_means) + b_means
                means = tf.identity(means, name="means")

                stds = tf.matmul(h, w_stds) + b_stds
                stds = tf.nn.softplus(stds) + 1e-5
                stds = tf.identity(stds, name="stds")

                normal_dist = tf.distributions.Normal(loc=means, scale=stds)
                self.normal_dists.append(normal_dist)

                action = tf.squeeze(normal_dist.sample(1))

                actions.append(action)

            self.actions = tf.stack(actions, name="actions")

        with tf.variable_scope("baseline", reuse=tf.AUTO_REUSE):
            self.baseline = tf.layers.dense(tf.reshape(H, (1, self.NUM_AGENTS * HIDDEN_VECTOR_LEN)), units=1,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.baseline = tf.squeeze(self.baseline)

            tf.summary.histogram("w_baseline", tf.get_variable("dense/kernel"))

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

            feed_dict = {self.observation: state, self.reward: reward, self.actions: actions}
            sess.run([self.train_op], feed_dict)

        self.ep_observations = []
        self.ep_actions = []
        self.ep_rewards = []


if __name__ == '__main__':
    tf.set_random_seed(42)

    NUM_AGENTS = 5
    VECTOR_OBS_LEN = 1
    OUTPUT_LEN = 1

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
