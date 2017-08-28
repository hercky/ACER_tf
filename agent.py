
import numpy as np
import tensorflow as tf
import sonnet as snt
import os
import time
import random
import sys
import matplotlib.pyplot as plt

import itertools
import collections
import random


import gym
from gym.wrappers import Monitor


from memory import Memory
from policy_net import PolicyNet
from advantage_net import AdvantageValueNet

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "distribution", "next_state"])



class Agent():
    
    def __init__(self, name = -1,
                 environment = None, 
                 global_counter = 0, 
                 average_actor_net = None, 
                 co_var = 0.3,
                 summary_writer = None,
                 saver = None,
                 optimizer = None,
                 flags = None):
        
        self.name = "acer_agent_" + name
        self.memory = Memory(5000) # each worker has its own memory
        
        # all the flags variables here 
        self.FLAGS = flags

        # for dumping info about this agent 
        self.file_dump = open("./dump/" + self.name + "_dump", 'w', 0)
        
        # average net copied
        self.average_actor_net = average_actor_net
        

        # if shared optimizer given use that or else create its own 
        if optimizer is None:
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.FLAGS.lr)
        else:
            self.optimizer = optimizer
        
        # env here 
        self.env = environment
        
        self.ACTION_DIM = self.env.action_space.shape[0]
        self.INPUT_DIM = self.env.observation_space.shape[0]

        
        # summary, saver, checkpointing
        self.summary_writer = summary_writer
        self.saver = saver
        
        if summary_writer is not None:
            self.checkpoint_path = os.path.abspath(os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))
        
        
        # diagonal co var for policy
        self.co_var = co_var
        
        # counter
        self.local_counter = itertools.count()
        self.global_counter = global_counter #next(self.global_counter)
        
        # loss function and optimizer in build graphs
        self.build_graph()
        
    def build_graph(self):
        """
        builds a local graph
        """
        # place holders for inputs here 
        HIDDEN_LAYER = self.FLAGS.feature_layer_size
        
        self.x_i = tf.placeholder(dtype=tf.float32, shape = (None, self.INPUT_DIM), name="x_i")
        self.a_i = tf.placeholder(dtype=tf.float32, shape = (None, self.ACTION_DIM), name = "a_i")
        self.q_opc = tf.placeholder(dtype=tf.float32, shape = (None, 1), name = "q_opc")
        self.q_ret = tf.placeholder(dtype=tf.float32, shape = (None, 1), name = "q_ret" )
        self.c = self.FLAGS.c # truncation threshold constant
        
        self.actor_net = PolicyNet(HIDDEN_LAYER, self.ACTION_DIM, name= self.name + "_actor", co_var = self.co_var)
        self.critic_net = AdvantageValueNet(HIDDEN_LAYER , name= self.name + "_critic")
        
        self.policy_xi_stats, self.policy_xi_dist = self.actor_net(self.x_i)
        
        self.val_xi, self.adv_xi_ai = self.critic_net(self.x_i, self.a_i, self.policy_xi_dist)
        
        #sample a' now
        self.a_i_ = tf.reshape(self.policy_xi_dist.sample(1), shape=[-1,self.ACTION_DIM])
        
        _, self.adv_xi_ai_ = self.critic_net(self.x_i, self.a_i_, self.policy_xi_dist) # val will be the same for 
        
        _, self.average_policy_xi_dist = self.average_actor_net(self.x_i) # can this be done better ?
        
        self.prob_a_i = tf.reshape(self.policy_xi_dist.prob(self.a_i),shape=[-1,1]) + 1e-8
        self.prob_a_i_ = tf.reshape(self.policy_xi_dist.prob(self.a_i_),shape=[-1,1]) + 1e-8
       
        self.log_prob_a_i = tf.log(self.prob_a_i)
        self.log_prob_a_i_ = tf.log(self.prob_a_i_)
        
        # for predicting 1-step a_i', p_i, p_i',
        self.u_i = tf.placeholder(dtype=tf.float32, shape = (None, self.ACTION_DIM))
        
        self.u_i_dist = tf.contrib.distributions.MultivariateNormalDiag(loc= self.u_i, 
                                                               scale_diag = tf.ones_like(self.u_i) * self.co_var)
        
        self.u_i_prob_a_i = tf.reshape(self.u_i_dist.prob(self.a_i),shape=[-1,1]) + 1e-8
        self.u_i_prob_a_i_ = tf.reshape(self.u_i_dist.prob(self.a_i_),shape=[-1,1]) + 1e-8
        
        self.p_i = tf.divide(self.prob_a_i, self.u_i_prob_a_i)
        self.p_i_ = tf.divide(self.prob_a_i_ , self.u_i_prob_a_i_)
        

        # take care of NaNs here, for importance sampling weights (might be an extra step)
        self.p_i = tf.where(tf.is_nan(self.p_i), tf.zeros_like(self.p_i), self.p_i)
        self.p_i_ = tf.where(tf.is_nan(self.p_i_), tf.zeros_like(self.p_i_), self.p_i_)

        self.c_i = tf.minimum(1. , tf.pow(self.p_i, 1.0/self.ACTION_DIM))
        
        
        # for verification about checking if params are getting synched
        self.local_actor_vars = self.actor_net.local_params()
        self.global_actor_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_actor')
        
        self.local_critic_vars = self.critic_net.local_params()
        self.global_critic_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global_critic')
        
        
        # Sync ops from global
        self.sync_local_actor_op = self.actor_net.update_local_params_op('global_actor') # global actor
        self.sync_local_critic_op = self.critic_net.update_local_params_op('global_critic')
        
        # soft update the average network
        self.soft_update_average_actor_op = self.average_actor_net.soft_update_from_target_params('global_actor',
                                                                                                  self.FLAGS.tau)
        
        #Get gradients from local network using local losses
        
        g1 = tf.reshape(tf.gradients((self.log_prob_a_i * (self.q_opc - self.val_xi)),self.policy_xi_stats,
                                     name=self.name+"g1_grads"), shape = [-1,self.ACTION_DIM])
        g2 = (self.adv_xi_ai_ - self.val_xi) * tf.reshape(tf.gradients((self.log_prob_a_i_), 
                                        self.policy_xi_stats, name=self.name+"g2_grads"), shape = [-1,self.ACTION_DIM])
        
        
        self.g = tf.minimum(self.c, self.p_i) * g1 + tf.nn.relu(1 - tf.divide(self.c , self.p_i_)) * g2
        
        
        self.k = tf.reshape(tf.gradients(tf.contrib.distributions.kl_divergence(self.average_policy_xi_dist, 
                                                self.policy_xi_dist), self.policy_xi_stats), shape = [-1,self.ACTION_DIM])
        
        
        
        self.kg = tf.reduce_sum( tf.multiply(self.g, self.k), 1, keep_dims=True)
        
        #print "kg", self.kg
        
        self.k2 = tf.reduce_sum( tf.multiply(self.k, self.k), 1, keep_dims=True)
        
        self.reg_g =  self.g - tf.maximum(tf.zeros_like(self.g), tf.divide((self.kg - self.FLAGS.delta), self.k2) ) * self.k
        
        # take gradients wrt to the local params
        self.actor_grads = tf.gradients(self.policy_xi_stats, self.local_actor_vars, 
                                        grad_ys= -self.reg_g, name="actor_grads")
        
        
        #for ti,tj in zip(self.actor_grads, self.global_actor_vars):
        #    print ti, "\n", tj , "\n", "==========="
        
        # apply local gradients to the global network
        self.actor_train_op = self.optimizer.apply_gradients(zip(self.actor_grads, self.global_actor_vars),
                                                             global_step=tf.train.get_global_step())
        
        
        # critic loss function and updates
        
        # take gradient wrt to local variables
        self.critic_loss_1 = ((self.q_ret - self.adv_xi_ai) ** 2.0) / 2.0
        
        # for predicting 1-step a_i', p_i, p_i',
        self.v_target = tf.placeholder(dtype=tf.float32, shape = (None, 1))
        
        #self.v_trunc = tf.minimum(self.p_i, 1.0) * (self.q_ret - self.adv_xi_ai) + self.val_xi
        self.critic_loss_2 = ((self.v_target - self.val_xi) ** 2.0) / 2.0
        
        self.critic_loss = self.critic_loss_1 + self.critic_loss_2
        
        #Apply local gradients to global network
        
        self.critic_grads = tf.gradients(self.critic_loss, self.local_critic_vars)
        
        self.critic_train_op = self.optimizer.apply_gradients(zip(self.critic_grads, self.global_critic_vars),
                                                             global_step=tf.train.get_global_step())
        
        # critic_summaries op
        critic_grads_summary = []
        for grad,var in zip(self.critic_grads, self.local_critic_vars):
            critic_grads_summary.append(tf.summary.histogram(var.name + '/gradient', grad))
            critic_grads_summary.append(tf.summary.histogram(var.name + '/weight', var))
        
        self.critic_summary_op = tf.summary.merge([
            tf.summary.scalar(self.name + "_critc_mean_loss_Q", tf.reduce_mean(self.critic_loss_1)),
            tf.summary.scalar(self.name + "_critc_mean_loss_V", tf.reduce_mean(self.critic_loss_2)),
            tf.summary.scalar(self.name + "_critc_sum_loss_Q", tf.reduce_sum(self.critic_loss_1)),
            tf.summary.scalar(self.name + "_critc_sum_loss_V", tf.reduce_sum(self.critic_loss_2)),
            tf.summary.scalar(self.name + "_critc_mean_loss", tf.reduce_mean(self.critic_loss)),
            tf.summary.scalar(self.name + "_critc_sum_loss", tf.reduce_sum(self.critic_loss)),
            tf.summary.histogram(self.name + "_val_target", self.v_target),
            tf.summary.histogram(self.name + "_val_pred", self.val_xi),
            tf.summary.histogram(self.name + "_Q_pred", self.adv_xi_ai),
            tf.summary.histogram(self.name + "_Q_ret", self.q_ret),
            tf.summary.histogram(self.name + "_Q_opc", self.q_opc),
        ] + critic_grads_summary)
        
        
        # actor summaries op

        actor_grads_summary = []
        for grad,var in zip(self.actor_grads, self.local_actor_vars):
            actor_grads_summary.append(tf.summary.histogram(var.name + '/gradient', grad))
            actor_grads_summary.append(tf.summary.histogram(var.name + '/weight', var))
        

        self.actor_summary_op = tf.summary.merge([
            tf.summary.scalar(self.name + "_actor_mean_loss_reg_g", tf.reduce_mean(self.reg_g)),
            tf.summary.scalar(self.name + "_actor_neg_mean_loss_reg_g", tf.reduce_mean(-self.reg_g)),
            tf.summary.scalar(self.name + "_actor_sum_loss_reg_g", tf.reduce_sum(self.reg_g)),
            tf.summary.scalar(self.name + "_actor_neg_sum_reg_g", tf.reduce_sum(-self.reg_g)),
            tf.summary.scalar(self.name + "_actor_sum_g", tf.reduce_sum(self.g)),
            tf.summary.scalar(self.name + "_actor_neg_sum_g", tf.reduce_sum(-self.g)),
            tf.summary.scalar(self.name + "_actor_mean_kl", tf.reduce_mean(self.k)),
            tf.summary.scalar(self.name + "_actor_sum_kl", tf.reduce_sum(self.k)),
            tf.summary.histogram(self.name + "_policy_stats", self.policy_xi_stats),
        ] + actor_grads_summary )
        
        
    def run(self, sess, coord) :
        """
        Main method, the ACER algorithm runs via this method

        """
        # use the prev session        
        with sess.as_default(), sess.graph.as_default():  
            # run stuff here
            try:
                # keep running the agents in a while loop
                while not coord.should_stop():
                    
                    # gather experiences
                    for i in range(self.FLAGS.pure_exploration_steps):
                        eps_reward, eps_len, local_t ,global_t = self.random_exploration_step(sess)
                        # use eps-greedy
                    
                    # 1 time current policy
                    for i in range(self.FLAGS.current_policy_steps):
                        eps_reward, eps_len, local_t ,global_t = self.current_policy_step(sess)
                        
                    # train off policy
                    for i in range(self.FLAGS.update_steps):
                        self.train_off_policy(sess)
                    
            except tf.errors.CancelledError:
                return

    def train_off_policy(self, sess):
        """
        ACER algorithm updates here 
        """
        
        # sync the local nets from the global
        sess.run([self.sync_local_actor_op, self.sync_local_critic_op])

        # sample trajectory from the replay memory
        traj = self.memory.get_trajectory(self.FLAGS.k_steps)
        k = len(traj)

        # empty list to store targets
        q_ret_list = []
        q_opc_list = []
        states = []
        actions = []
        mu_dist = []
        val_s = []
        
        # if last episode is not terminal state, use value function to get an estimate
        Q_ret = 0.0
        if not traj[-1].done :
            Q_ret = sess.run([self.val_xi], feed_dict= { 
                self.x_i :np.reshape(traj[-1].next_state, (1,-1))} )[0][0,0]
            
        Q_opc = Q_ret 
        
        # reverse loop
        for transition in traj[::-1]:
            Q_ret = transition.reward +  self.FLAGS.gamma * Q_ret
            Q_opc = transition.reward +  self.FLAGS.gamma * Q_opc
            
            #
            x_t = np.reshape(transition.state, (1,-1))
            a_t = np.reshape(transition.action, (1,-1))
            u_t = np.reshape(transition.distribution, (1,-1))
            
            # add to minibatch
            q_ret_list.append(Q_ret)
            q_opc_list.append(Q_opc)
            states.append(x_t)
            actions.append(a_t)
            mu_dist.append(u_t)
            
            
            # get estimates from existing function approximators
            v_t, c_t, p_t, q_t = sess.run([self.val_xi, self.c_i, self.p_i, self.adv_xi_ai], 
                        feed_dict = {self.x_i : x_t, 
                                     self.a_i: a_t, 
                                     self.u_i: u_t} )
            
            # add the target V_pi
            val_s.append((min(p_t[0,0],1.0) * (Q_ret - q_t[0,0])) + v_t[0,0] )
            
            # update again
            Q_ret  = c_t[0,0] * (Q_ret - q_t[0,0]) + v_t[0,0] 
            Q_opc = (Q_opc - q_t[0,0]) + v_t[0,0]
            
        
        # create mini-batch here
        opt_feed_dict = {
            self.x_i : np.asarray(states).reshape(-1, self.INPUT_DIM),
            self.a_i : np.asarray(actions).reshape(-1, self.ACTION_DIM),
            self.q_opc : np.asarray(q_opc_list).reshape(-1, 1), 
            self.q_ret : np.asarray(q_ret_list).reshape(-1, 1), 
            self.u_i : np.asarray(mu_dist).reshape(-1,self.ACTION_DIM),
            self.v_target: np.asarray(val_s).reshape(-1,1)
        }
            
        # Train the global estimators using local gradients
        _,_, global_step, critic_summaries, actor_summaries = sess.run([self.actor_train_op, 
                                                       self.critic_train_op, 
                                                       tf.contrib.framework.get_global_step(),
                                                       self.critic_summary_op, 
                                                       self.actor_summary_op ], 
                                                      feed_dict=opt_feed_dict)
        
        
        # Write summaries
        if self.summary_writer is not None:
            self.summary_writer.add_summary(critic_summaries, global_step)
            self.summary_writer.add_summary(actor_summaries, global_step)
            self.summary_writer.flush()
        
        
        # update the average policy network
        _ = sess.run([self.soft_update_average_actor_op])
        
        # that's it 
        
        
    def random_exploration_step(self, sess):
        """
        follow a random uniform policy to gather experiences and add them to the replay memory
        """
        episode_reward = 0.0
        episode_len = 0 #num of action
        
        # random policy
        random_policy = np.zeros((1,self.ACTION_DIM))
        
        #for each episode reset first
        state = self.env.reset()
        for t in range(self.FLAGS.max_episode_len):
            action = self.env.action_space.sample() # random action
            
            next_state, reward, done, info = self.env.step(action) # next state, reward, terminal
            
            # insert this in memory with a uniform distribution over actions
            
            self.memory.add(Transition(state=state, action=action, 
                                       reward=reward, done=done, 
                                       distribution=random_policy, next_state = next_state))
            
            # accumulate rewards
            episode_reward += reward
            episode_len += 1 
            
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)
            
            
            # update the state 
            state = next_state 
            
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
        return episode_reward, episode_len, local_t, global_t
        
        
    def current_policy_step(self, sess, add_to_mem = True):
        """
        follow the current policy network and gather trajectories and update them in the replay memory
        
        return the reward for this epiosde here
        # plot the reward over the trajectories
        """
        
        episode_reward = 0.0
        episode_len = 0 #num of action
        
        #for each episode reset first
        state = self.env.reset()
        
        for t in range(self.FLAGS.max_episode_len):
            
            # take action according to current policy
            
            action, policy_stats = sess.run([self.a_i_, self.policy_xi_stats], 
                                            feed_dict={self.x_i : np.array([state])})
            
            action = np.reshape(action, (self.ACTION_DIM,))
            
            next_state, reward, done, info = self.env.step(action) # next state, reward, terminal
            
            # insert this in memory with a uniform distribution over actions
            if add_to_mem:  # can also remove this and still work/optimization
                self.memory.add(Transition(state=state, action=action, 
                                       reward=reward, done=done, 
                                       distribution=policy_stats, next_state=next_state))
            
            # accumulate rewards
            episode_reward += reward
            episode_len += 1 
            
            local_t = next(self.local_counter)
            global_t = next(self.global_counter)
            
            # update the state 
            state = next_state 
            
            if done:
                # print("Episode finished after {} timesteps".format(t+1))
                break
        
        return episode_reward, episode_len, local_t, global_t

        
    def evaluate_policy(self, sess,  eval_every = 3600, coord = None):
        """
        follow the current policy network and gather trajectories and update them in the replay memory
        
        return the reward for this epiosde here
        # plot the reward over the trajectories
        """
        
        self.video_dir = os.path.join(self.summary_writer.get_logdir(), "../videos")
        self.video_dir = os.path.abspath(self.video_dir)
        
        try:
            os.makedirs(self.video_dir)
        except Exception:
            pass
        
        self.env._max_episode_steps = self.FLAGS.max_episode_len
        self.env = Monitor(self.env, directory=self.video_dir, video_callable=lambda x: True, resume=True)
        
        with sess.as_default(), sess.graph.as_default():  
            # run stuff here
            
            try:
            
                while not coord.should_stop():
                    
                    # sync the actor
                    global_step, _ = sess.run([tf.contrib.framework.get_global_step(), self.sync_local_actor_op])
        
                    #for each episode reset first        
                    eps_reward, eps_len, _ , global_t = self.current_policy_step(sess, add_to_mem=False)
                    
                    
                    # Add summaries
                    if self.summary_writer is not None:
                        episode_summary = tf.Summary()
                        episode_summary.value.add(simple_value=eps_reward, tag=self.name + "/total_reward")
                        episode_summary.value.add(simple_value=eps_len, tag=self.name+"/episode_length")
                        self.summary_writer.add_summary(episode_summary, global_step)
                        
                        episode_summary_frame = tf.Summary()
                        episode_summary_frame.value.add(simple_value=eps_reward, tag=self.name + "/frame/total_reward")
                        episode_summary_frame.value.add(simple_value=eps_len, tag=self.name+"/frame/episode_length")
                        self.summary_writer.add_summary(episode_summary_frame, global_t)
                        
                        self.summary_writer.flush()
                        

                    if self.saver is not None:
                        self.saver.save(sess, self.checkpoint_path)

                    tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}".format(global_step, eps_reward, eps_len))
                    tf.logging.info("Total steps taken so far: {}".format(self.global_counter))
                    
                    # # Sleep until next evaluation cycle
                    time.sleep(eval_every)
                    
                    # for stopping once 
#                     coord.request_stop()
#                     return
            except tf.errors.CancelledError:
                return
        
