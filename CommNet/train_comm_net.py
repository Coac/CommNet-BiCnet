"""
Implementation of DDPG - Deep Deterministic Policy Gradient https://github.com/pemami4911/deep-rl
Modified by Coac for CommNet implementation https://github.com/Coac/CommNet-BiCnet
"""
import argparse
import pprint as pp
from datetime import datetime

import numpy as np
import tensorflow as tf
from comm_net import CommNet
from guessing_sum_env import *
from replay_buffer import ReplayBuffer

HIDDEN_VECTOR_LEN = 1
NUM_AGENTS = 2
VECTOR_OBS_LEN = 1
OUTPUT_LEN = 1


# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        self.inputs, self.out = self.create_actor_network("actor_network")
        self.network_params = tf.trainable_variables()

        self.target_inputs, self.target_out = self.create_actor_network("target_actor_network")
        self.target_network_params = tf.trainable_variables()[
                                     len(self.network_params):]

        with tf.name_scope("actor_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                      tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.action_gradient = tf.placeholder(tf.float32, (None, self.a_dim[0], self.a_dim[1]), name="action_gradient")

        with tf.name_scope("actor_gradients"):
            self.unnormalized_actor_gradients = tf.gradients(self.out, self.network_params, -self.action_gradient)
            self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        self.optimize = tf.train.AdamOptimizer(self.learning_rate)
        self.optimize = self.optimize.apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="actor_inputs")
        out = CommNet.actor_build_network(name, inputs)
        return inputs, out

    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: action_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):
    """
    Input to the network is the state and action, output is Q(s,a).
    The action must be obtained from the output of the Actor network.

    """

    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        self.inputs, self.action, self.out = self.create_critic_network("critic_network")
        self.network_params = tf.trainable_variables()[num_actor_vars:]

        self.target_inputs, self.target_action, self.target_out = self.create_critic_network("target_critic_network")
        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        with tf.name_scope("critic_update_target_network_params"):
            self.update_target_network_params = \
                [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau)
                                                      + tf.multiply(self.target_network_params[i], 1. - self.tau))
                 for i in range(len(self.target_network_params))]

        self.predicted_q_value = tf.placeholder(tf.float32, (None, 1), name="predicted_q_value")

        self.loss = tf.losses.mean_squared_error(self.predicted_q_value, self.out)

        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action, name="action_grads")

    def create_critic_network(self, name):
        inputs = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, VECTOR_OBS_LEN), name="critic_inputs")
        action = tf.placeholder(tf.float32, shape=(None, NUM_AGENTS, OUTPUT_LEN), name="critic_action")

        out = CommNet.critic_build_network(name, inputs, action)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize, self.loss], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)


# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0., name="episode_reward")
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0., name="episode_ave_max_q")
    tf.summary.scalar("Qmax Value", episode_ave_max_q)
    loss = tf.Variable(0., name="critic_loss")
    tf.summary.scalar("Critic_loss", loss)

    summary_vars = [episode_reward, episode_ave_max_q, loss]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars


# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor, critic):
    summary_ops, summary_vars = build_summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(args['summary_dir'] +  " actor_lr" + str(args['actor_lr']) + " critic_lr" + str(args["critic_lr"]), sess.graph)

    actor.update_target_network()
    critic.update_target_network()

    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):
        state = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):
            action = actor.predict([state])[0]

            state2, reward, done, info = env.step(action)
            reward = np.sum(reward)

            replay_buffer.add(state, action, reward, done, state2)

            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # TODO
                # Calculate targets
                # target_q = critic.predict_target(
                #     s2_batch, actor.predict_target(s2_batch))

                target_q = tf.zeros((1))

                # Update the critic given the targets
                predicted_q_value, _, loss = critic.train(s_batch, a_batch,
                                                          np.reshape(r_batch, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                actor.update_target_network()
                critic.update_target_network()

                replay_buffer.clear()

                # Log
                summary_str = sess.run(summary_ops, feed_dict={
                    summary_vars[0]: np.mean(r_batch),
                    summary_vars[1]: ep_ave_max_q / float(j + 1),
                    summary_vars[2]: loss
                })

                writer.add_summary(summary_str, i)
                writer.flush()

                print('| Reward: {:.4f} | Episode: {:d} | Qmax: {:.4f}'.format(np.mean(r_batch),
                                                                               i, (ep_ave_max_q / float(j + 1))))

            state = state2
            ep_reward += reward

            if done:
                break


def main(args):
    args = parse_arg(args or None)

    tf.reset_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        env = GuessingSumEnv(NUM_AGENTS)
        env.seed(0)

        np.random.seed(int(args['random_seed']))
        tf.set_random_seed(int(args['random_seed']))
        env.seed(int(args['random_seed']))

        state_dim = (NUM_AGENTS, VECTOR_OBS_LEN)
        action_dim = (NUM_AGENTS, OUTPUT_LEN)

        actor = ActorNetwork(sess, state_dim, action_dim,
                             float(args['actor_lr']), float(args['tau']),
                             int(args['minibatch_size']))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(args['critic_lr']), float(args['tau']),
                               float(args['gamma']),
                               actor.get_num_trainable_vars())

        train(sess, env, args, actor, critic)


def parse_arg(args):
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.01)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.15)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=1024)

    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=9999999999999)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info',
                        default="summaries/" + datetime.now().strftime('%d-%m-%y %H%M'))

    args = vars(parser.parse_args(args))

    pp.pprint(args)

    return args


if __name__ == '__main__':
    main()
