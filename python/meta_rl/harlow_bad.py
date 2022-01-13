# Copyright 2016 Google Inc.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import random
import numpy as np
import six

import deepmind_lab

import os
import tensorflow as tf
from meta_rl.ac_network import AC_Network
from meta_rl.worker import Worker

from datetime import datetime

import threading
import multiprocessing

import collections
import statistics
import tqdm

from matplotlib import pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.keras import layers
from typing import Any, List, Sequence, Tuple


from skimage import data
from skimage.color import rgb2gray

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

DATASET_SIZE = 2

# Padding
WIDTH_PAD  = 18 # 19 if we dont want 1 pixel surrounding image
HEIGHT_PAD = 30 # 31 if we dont want 1 pixel surrounding image

config = {
    'fps': str(60),
    'width': str(84),
    'height': str(84)
}

# Create the environment
env = deepmind_lab.Lab('contributed/psychlab/harlow', ['RGB_INTERLEAVED'], config=config)

# Set seed for experiment reproducibility
seed = 42
#tf.random.set_seed(seed)
#np.random.seed(seed)

# Small epsilon value for stabilizing division operations
eps = np.finfo(np.float32).eps.item()

class ActorCritic(tf.keras.Model):
  """Combined actor-critic network."""

  def __init__(
      self,
      num_actions: int,
      num_hidden_units: int):
    """Initialize."""
    super().__init__()

    self.conv_1 = layers.Conv2D(32, (3, 3), activation='relu', input_shape=(84, 84, 3))
    self.max_pool_1 = layers.MaxPooling2D((2, 2))
    self.conv_2 = layers.Conv2D(64, (3, 3), activation='relu')
    self.max_pool_2 = layers.MaxPooling2D((2, 2))
    self.conv_3 = layers.Conv2D(64, (3, 3), activation='relu')
    self.flatten = layers.Flatten()
    self.dense_1 = layers.Dense(1024, activation='relu')
    self.dense_2 = layers.Dense(512)
    self.embedding = layers.Embedding(input_dim=512, output_dim=num_hidden_units)
    self.lstm = layers.LSTM(num_hidden_units, activation="tanh")
    self.actor = layers.Dense(num_actions)
    self.critic = layers.Dense(1)

  def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    x = self.conv_1(inputs)
    x = self.max_pool_1(x)
    x = self.conv_2(x)
    x = self.max_pool_2(x)
    x = self.conv_3(x)
    x = self.flatten(x)
    x = self.dense_1(x)
    x = self.dense_2(x)
    x = self.embedding(x)
    x = self.lstm(x)
    x = ops.convert_to_tensor(x)
    #print(x)
    #print(self.actor(x))
    #print(self.critic(x))
    return self.actor(x), self.critic(x)

# Wrap OpenAI Gym's `env.step` call as an operation in a TensorFlow function.
# This would allow it to be included in a callable TensorFlow graph.

def _action(*entries):
  return np.array(entries, dtype=np.intc)

ACTIONS = {
  'look_left': _action(-5, 0, 0, 0, 0, 0, 0),
  'look_right': _action(5, 0, 0, 0, 0, 0, 0),
  'no-ops': _action(0, 0, 0, 0, 0, 0, 0)
}

ACTION_LIST = list(six.viewvalues(ACTIONS))

def env_step(action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Returns state, reward and done flag given an action."""
  done = not env.is_running()
  reward = env.step(ACTION_LIST[action])
  state = np.float32(env.observations()['RGB_INTERLEAVED'])
  return (state.astype(np.float32),
          np.array(reward, np.int32),
          np.array(done, np.int32))

def tf_env_step(action: tf.Tensor) -> List[tf.Tensor]:
  return tf.numpy_function(env_step, [action],
                           [tf.float32, tf.int32, tf.int32])

def run_episode(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    max_steps: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Runs a single episode to collect training data."""

  action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
  rewards = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

  initial_state_shape = initial_state.shape
  state = initial_state

  for t in tf.range(max_steps):
    # Convert state into a batched tensor (batch size = 1)
    state = tf.expand_dims(state, 0)

    # Run the model and to get action probabilities and critic value
    action_logits_t, value = model(state)

    # Sample next action from the action probability distribution
    action = tf.random.categorical(action_logits_t, 1)[0, 0]
    action_probs_t = tf.nn.softmax(action_logits_t)

    # Store critic values
    values = values.write(t, tf.squeeze(value))

    # Store log probability of the action chosen
    action_probs = action_probs.write(t, action_probs_t[0, action])

    # Apply action to the environment to get next state and reward
    state, reward, done = tf_env_step(action)
    print(state)
    print(reward)
    print(done)
    state.set_shape(initial_state_shape)

    # Store reward
    rewards = rewards.write(t, reward)

    if tf.cast(done, tf.bool):
      break

  action_probs = action_probs.stack()
  values = values.stack()
  rewards = rewards.stack()

  return action_probs, values, rewards

def get_expected_return(
    rewards: tf.Tensor,
    gamma: float,
    standardize: bool = True) -> tf.Tensor:
  """Compute expected returns per timestep."""

  n = tf.shape(rewards)[0]
  returns = tf.TensorArray(dtype=tf.float32, size=n)

  # Start from the end of `rewards` and accumulate reward sums
  # into the `returns` array
  rewards = tf.cast(rewards[::-1], dtype=tf.float32)
  discounted_sum = tf.constant(0.0)
  discounted_sum_shape = discounted_sum.shape
  for i in tf.range(n):
    reward = rewards[i]
    discounted_sum = reward + gamma * discounted_sum
    discounted_sum.set_shape(discounted_sum_shape)
    returns = returns.write(i, discounted_sum)
  returns = returns.stack()[::-1]

  if standardize:
    returns = ((returns - tf.math.reduce_mean(returns)) /
               (tf.math.reduce_std(returns) + eps))

  return returns

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)

def compute_loss(
    action_probs: tf.Tensor,
    values: tf.Tensor,
    returns: tf.Tensor) -> tf.Tensor:
  """Computes the combined actor-critic loss."""

  advantage = returns - values

  action_log_probs = tf.math.log(action_probs)
  actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

  critic_loss = huber_loss(values, returns)

  return actor_loss + critic_loss

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

@tf.function
def train_step(
    initial_state: tf.Tensor,
    model: tf.keras.Model,
    optimizer: tf.keras.optimizers.Optimizer,
    gamma: float,
    max_steps_per_episode: int) -> tf.Tensor:
  """Runs a model training step."""

  with tf.GradientTape() as tape:
    print(initial_state)
    # Run the model for one episode to collect training data
    action_probs, values, rewards = run_episode(
        initial_state, model, max_steps_per_episode)

    # Calculate expected returns
    returns = get_expected_return(rewards, gamma)

    # Convert training data to appropriate TF tensor shapes
    action_probs, values, returns = [
        tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

    # Calculating loss values to update our network
    loss = compute_loss(action_probs, values, returns)

  # Compute the gradients from the loss
  grads = tape.gradient(loss, model.trainable_variables)

  # Apply the gradients to the model's parameters
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  episode_reward = tf.math.reduce_sum(rewards)

  return episode_reward

def run(length, width, height, fps, level, record, demo, demofiles, video):
  """Spins up an environment and runs the random agent."""

  if record:
    config['record'] = record
  if demo:
    config['demo'] = demo
  if demofiles:
    config['demofiles'] = demofiles
  if video:
    config['video'] = video

  dir_name = "/home/damien/Documents/GitHub/lab/python/meta_rl/meta_rl/train_000"#datetime.now().strftime("%m%d-%H%M%S")

  # initialize the directories' names to save the models for this particular seed
  model_path = dir_name+'/model_' + str(seed)
  frame_path = dir_name+'/frames_' + str(seed)
  plot_path = dir_name+'/plots_' + str(seed)
  #load_model_path = "/home/damien/Documents/GitHub/lab/python/meta_rl/meta_rl/results/biorxiv/final/model_" + str(seed_nb) + "/model-20000"
  load_model_path = "/home/damien/Documents/GitHub/lab/python/meta_rl/meta_rl/train_000/model_0/model-20"
  # create the directories
  if not os.path.exists(model_path):
    os.makedirs(model_path)
  if not os.path.exists(frame_path):
    os.makedirs(frame_path)
  if not os.path.exists(plot_path):
    os.makedirs(plot_path)

  num_actions = len(ACTION_LIST)  # 3
  num_hidden_units = 256

  model = ActorCritic(num_actions, num_hidden_units)

  min_episodes_criterion = 100
  max_episodes = 10000
  max_steps_per_episode = 3600

  reward_threshold = 31
  running_reward = 0

  # Discount factor for future rewards
  gamma = 0.91

  # Keep last episodes reward
  episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

  with tqdm.trange(max_episodes) as t:
      for i in t:
        env.reset()
        initial_state = np.float32(env.observations()['RGB_INTERLEAVED'])
        episode_reward = int(train_step(initial_state, model, optimizer, gamma, max_steps_per_episode))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_description(f'Episode {i}')
        t.set_postfix(episode_reward=episode_reward, running_reward=running_reward)

        # Show average episode reward every 10 episodes
        if i % 10 == 0:
          pass # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
          break

  print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--length', type=int, default=1000,
                      help='Number of steps to run the agent')
  parser.add_argument('--width', type=int, default=84,
                      help='Horizontal size of the observations')
  parser.add_argument('--height', type=int, default=84,
                      help='Vertical size of the observations')
  parser.add_argument('--fps', type=int, default=60,
                      help='Number of frames per second')
  parser.add_argument('--runfiles_path', type=str, default=None,
                      help='Set the runfiles path to find DeepMind Lab data')
  parser.add_argument('--level_script', type=str,
                      default='contributed/psychlab/harlow',
                      help='The environment level script to load')
  parser.add_argument('--record', type=str, default=None,
                      help='Record the run to a demo file')
  parser.add_argument('--demo', type=str, default=None,
                      help='Play back a recorded demo file')
  parser.add_argument('--demofiles', type=str, default=None,
                      help='Directory for demo files')
  parser.add_argument('--video', type=str, default=None,
                      help='Record the demo run as a video')

  args = parser.parse_args()
  if args.runfiles_path:
    deepmind_lab.set_runfiles_path(args.runfiles_path)
  run(args.length, args.width, args.height, args.fps, args.level_script,
      args.record, args.demo, args.demofiles, args.video)
