import numpy as np
import tensorflow as tf
import sonnet as snt
import os
import time
import random
import sys
import matplotlib.pyplot as plt


import threading
import multiprocessing

# reset stuff
tf.reset_default_graph()


import time
import pylab
import seaborn as sns


import itertools
import collections
import random

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "done", "distribution", "next_state"])


import shutil


from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from memory import Memory
from policy_net import PolicyNet
from advantage_net import AdvantageValueNet
from agent import Agent 



tf.flags.DEFINE_string("model_dir", "./model/", "Directory to write Tensorboard summaries and videos to.")
tf.flags.DEFINE_string("env", "InvertedPendulum-v1", "Name of gym Mujoco environment, e.g. InvertedPendulum-v1")
tf.flags.DEFINE_integer("k_steps", 50, "Number of k-step returns.")
tf.flags.DEFINE_integer("max_episode_len", 500, "Maximum episode length.")
tf.flags.DEFINE_integer("eval_every_sec", 60, "Evaluate the policy every N seconds")
tf.flags.DEFINE_boolean("reset", True, "If set, delete the existing model directory and start training from scratch.")
tf.flags.DEFINE_integer("feature_layer_size", 200, "Num of units in the hidden layer for feature processing")
tf.flags.DEFINE_integer("num_agents", 1, "Number of threads to run. ")
tf.flags.DEFINE_float("co_var", 0.3, "Diagonal covariance for the multi variate normal policy.")
tf.flags.DEFINE_float("delta", 1.0, "Delta as defined in ACER for TRPO.")
tf.flags.DEFINE_float("lr", 1e-4, "Learning rate for the shared optimzer")
tf.flags.DEFINE_float("c", 5.0, "Importance sampling truncated ratio")
tf.flags.DEFINE_float("gamma", 0.99, "Discount factor")
tf.flags.DEFINE_float("tau", 0.995, "Soft update rule for average network params")
tf.flags.DEFINE_integer("pure_exploration_steps", 0, "Number of pure random policy steps to take  ")
tf.flags.DEFINE_integer("current_policy_steps", 4, "Number of current policy steps to take  ")
tf.flags.DEFINE_integer("update_steps", 1, "Number of off policy updates to perform per epoch  ")



FLAGS = tf.flags.FLAGS


import gym
from gym.wrappers import Monitor


def make_env(env_name = 'Swimmer-v1'):
    """
    create a copy of the environment and return that back
    """
    env = gym.make(FLAGS.env)
    env.seed(0)
    
    return env


MODEL_DIR = FLAGS.model_dir
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")

# get these from envs

ENV_NAME = FLAGS.env
HIDDEN_LAYER = FLAGS.feature_layer_size # feature hidden layer

CO_VAR = FLAGS.co_var
DELTA = FLAGS.delta
LEARNING_RATE = FLAGS.lr
TRUNCATION_THRESHOLD_C = FLAGS.c
DISCOUNT_FACTOR = FLAGS.gamma
TAU = FLAGS.tau

K_STEPS = FLAGS.k_steps
MAX_EPISODE_LEN = FLAGS.max_episode_len

EXPLOITATION_STEPS = FLAGS.update_steps
EVAL_EVERY_SEC = FLAGS.eval_every_sec # eval every 15 min 

PURE_EXPLORATION_STEPS = FLAGS.pure_exploration_steps
PREV_POLICY_STEPS = FLAGS.current_policy_steps



NUM_AGENTS = FLAGS.num_agents

# the driver program here 
tf.reset_default_graph()

BATCH_SIZE = None

RESET_DIR = FLAGS.reset

# Optionally empty model directory
if RESET_DIR:
    shutil.rmtree(MODEL_DIR, ignore_errors=True)

if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# Environment details here

env = make_env(env_name=ENV_NAME)

ACTION_DIM = env.action_space.shape[0]
INPUT_DIM = env.observation_space.shape[0]


# disable GPU memory usuage here
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# create the summary writer here
summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))


# to run stuff on cpu
with tf.device("/cpu:0"): 
    
    # Keeps track of the number of updates we've performed
    global_step = tf.Variable(0, name="global_step", trainable=False)
    
    
    global_actor_net = PolicyNet(HIDDEN_LAYER, ACTION_DIM, name="global_actor")
    global_critic_net = AdvantageValueNet(HIDDEN_LAYER, name="global_critic")
    # connecting stuff
    tmp_x = tf.placeholder(dtype=tf.float32, shape = (BATCH_SIZE, INPUT_DIM) , name ="tmp_x")
    tmp_a = tf.placeholder(dtype=tf.float32, shape = (BATCH_SIZE, ACTION_DIM), name ="tmp_a")
    
    global_average_actor_net = PolicyNet(HIDDEN_LAYER, ACTION_DIM, name="global_Average_actor")
    
    _, tmp_policy = global_actor_net(tmp_x)
    _ = global_critic_net(tmp_x, tmp_a,  tmp_policy)
    _ = global_average_actor_net(tmp_x)
    
    # optimzer with learning rate, alternate set it to None so each agent can create it own optimse
    optimizer = tf.train.RMSPropOptimizer(learning_rate=LEARNING_RATE) 
    
    
    # Global step iterator
    global_counter = itertools.count()
    
    
    
    num_agents = NUM_AGENTS # Set workers ot number of available CPU threads/ multiprocessing.cpu_count()
    agents = []
    # Create workers
    for i in range(num_agents):
        # We only write summaries in one of the workers because they're
        # pretty much identical and writing them on all workers
        # would be a waste of space
        agent_summary_writer = None
        if i == 0:
            agent_summary_writer = summary_writer
            
        agents.append(Agent(name=str(i),
                            environment=make_env(env_name=ENV_NAME), 
                            global_counter=global_counter, 
                            average_actor_net=global_average_actor_net, 
                            optimizer=optimizer, 
                            summary_writer = agent_summary_writer,
                            co_var=CO_VAR,
                            flags = FLAGS))
        
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.0, max_to_keep=10)
    
    # separate agent for keeping track of the videos as well as plotting to tensorboard
    policy_monitor_agent = Agent(name= "policy_monitor",
                            environment= make_env(env_name=ENV_NAME), 
                            global_counter= itertools.count(), # empty counter 
                            average_actor_net= global_average_actor_net, 
                            optimizer= None,  # doesn't matter
                            co_var= CO_VAR, 
                            summary_writer = summary_writer,
                            saver = saver,
                            flags = FLAGS)

#with tf.train.MonitoredSession() as sess:
with tf.Session() as sess:
    coord = tf.train.Coordinator() # might be cause for potential bug here for asynchronous stuff
    sess.run(tf.global_variables_initializer())
    
    latest_checkpoint = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if latest_checkpoint:
        print("Loading model checkpoint: {}".format(latest_checkpoint))
        saver.restore(sess, latest_checkpoint)
    

    # Run the agents
    agent_threads = []
    for agent in agents:
        agent_run = lambda: agent.run(sess, coord)
        t = threading.Thread(target=(agent_run))
        t.start()
        agent_threads.append(t)
        
    # Start a thread for policy eval task
    monitor_thread = threading.Thread(target=lambda: policy_monitor_agent.evaluate_policy(sess,EVAL_EVERY_SEC, coord))
    monitor_thread.start()

        
    coord.join(agent_threads)
        