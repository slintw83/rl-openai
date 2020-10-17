import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy, epsilon_greedy_policy, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

from Environment import BreakoutEnv

env_name = 'Breakout-v0'

train_py_env = BreakoutEnv(suite_gym.load(env_name))
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = BreakoutEnv(suite_gym.load(env_name))
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

learning_rate = 1e-5  # @param {type:"number"}
replay_buffer_max_length = 10000  # @param {type:"integer"}

num_iterations = 10000 # @param {type:"integer"}

initial_collect_steps = 2000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}

batch_size = 32  # @param {type:"integer"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}

conv_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
fc_layer_params = (512,)
q_net = q_network.QNetwork(train_env.observation_spec(), train_env.action_spec(), fc_layer_params=fc_layer_params, conv_layer_params=conv_layer_params, kernel_initializer=tf.initializers.GlorotNormal)

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

#loss_func = common.element_wise_squared_loss
loss_func = common.element_wise_huber_loss
agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer, 
    td_errors_loss_fn=loss_func, 
    train_step_counter=train_step_counter)

agent.initialize()

def decay_epsilon():
    ep = 1.0
    epsilon_min = 0.1
    epsilon_change = (ep - epsilon_min) / 500000
    def decay():
        nonlocal ep
        e = ep
        ep = max(ep - epsilon_change, epsilon_min)
        return e
    return decay

eval_policy = agent.policy
collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(agent.collect_policy.wrapped_policy, epsilon=decay_epsilon())

# Replay buffer collection
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

replay_observer = [replay_buffer.add_batch]

collect_op = dynamic_step_driver.DynamicStepDriver(
  train_env,
  collect_policy,
  observers=replay_observer,
  num_steps=initial_collect_steps)
collect_op.run()

dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)
iterator = iter(dataset)

# Trainning
metric = tf_metrics.AverageReturnMetric()
eval_observer = [metric]
eval_op = dynamic_step_driver.DynamicStepDriver(
    eval_env,
    eval_policy,
    observers=eval_observer,
    num_steps=num_eval_episodes
)

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
eval_op.run()
returns = [metric.result().numpy()]
print('Initial average reward = {0}'.format(returns[0]))


for _ in range(num_iterations):
    # Collect few steps into the replay buffer
    collect_op.run(maximum_iterations=1)

    experience, _ = next(iterator)
    train_loss = agent.train(experience).loss

    step = agent.train_step_counter.numpy()
    
    if step % log_interval == 0:
        print('step = {0}: loss = {1}'.format(step, train_loss))

    if step % eval_interval == 0:
        eval_op.run()
        avg_return = metric.result().numpy()
        print('step = {0}: Average Return = {1}'.format(step, avg_return))
        returns.append(avg_return)


def embed_mp4(filename):
  """Embeds an mp4 file in the notebook."""
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)

def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
  return embed_mp4(filename)

random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

create_policy_eval_video(random_policy, "trained-agent-random")
create_policy_eval_video(eval_policy, "trained-agent-first")