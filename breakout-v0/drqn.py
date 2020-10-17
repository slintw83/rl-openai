import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_rnn_network
from tf_agents.policies import random_tf_policy, epsilon_greedy_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

from Environment import BreakoutEnv

learning_rate = 1e-3  # @param {type:"number"}

replay_buffer_max_length = 100000  # @param {type:"integer"}
initial_collect_steps = 50000  # @param {type:"integer"} 

batch_size = 64  # @param {type:"integer"}

num_eval_episodes = 100  # @param {type:"integer"}
num_episodes = 250 # @param {type:"integer"}
log_interval = 200  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


env_name = 'Breakout-v0'
train_py_env = BreakoutEnv(suite_gym.load(env_name))
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

eval_py_env = BreakoutEnv(suite_gym.load(env_name))
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

conv_layer_params = [(32, 8, 4), (64, 4, 2)]
input_fc_layer_params = (256,)
output_fc_layer_params = (256,)
lstm_size = [256]

q_net = q_rnn_network.QRnnNetwork(
                            train_env.observation_spec(),
                            train_env.action_spec(),
                            input_fc_layer_params=input_fc_layer_params,
                            output_fc_layer_params=output_fc_layer_params,
                            conv_layer_params=conv_layer_params,
                            lstm_size=lstm_size)

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
loss_func = common.element_wise_huber_loss
train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
                train_env.time_step_spec(),
                train_env.action_spec(),
                q_network=q_net,
                optimizer=optimizer, 
                td_errors_loss_fn=loss_func, 
                train_step_counter=train_step_counter)
agent.initialize()

def decaying_epsilon():
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_change = (epsilon - epsilon_min) / 50000
    decay_step = 0
    def decay():
        nonlocal epsilon
        nonlocal decay_step
        _epsilon = epsilon
        epsilon = max(epsilon - epsilon_change, epsilon_min)
        decay_step += 1
        if decay_step % 500 == 0:
            print('Decaying epsilon from {0} to {1}'.format(_epsilon, epsilon))
        return _epsilon
    return decay

eval_policy = agent.policy  # Greedy policy
# collect_policy = agent.collect_policy  # Epsilon-greedy policy
collect_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(agent.collect_policy.wrapped_policy, epsilon=decaying_epsilon())
random_policy = random_tf_policy.RandomTFPolicy(
                                        action_spec=collect_policy.action_spec,
                                        time_step_spec=collect_policy.time_step_spec) # Random policy

# Replay buffer collection
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                        data_spec=agent.collect_data_spec,
                        batch_size=train_env.batch_size,
                        max_length=replay_buffer_max_length)

replay_observer = [replay_buffer.add_batch]

collect_op = dynamic_step_driver.DynamicStepDriver(
                                    train_env,
                                    random_policy,
                                    observers=replay_observer,
                                    num_steps=initial_collect_steps)
collect_op.run()

# Change collect op to use e-greedy policy
collect_op = dynamic_step_driver.DynamicStepDriver(
                                    train_env,
                                    collect_policy,
                                    observers=replay_observer)

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
                                    num_steps=num_eval_episodes)

# Evaluate the agent's policy once before training.
eval_op.run()
returns = [metric.result().numpy()]
print('Initial average reward = {0}\n'.format(returns[0]))

agent.train = common.function(agent.train)
agent.train_step_counter.assign(0)


for ep in range(num_episodes):
    done = False
    ps = None

    print('Starting episode #{0}'.format(ep))
    while not done:
        # Collect few steps into the replay buffer
        ts, ps = collect_op.run(maximum_iterations=1, policy_state=ps)
        done = ts.is_last()
        if done:
            print('Episode #{0} reward = {1}'.format(ep, ts.reward[0]))
            returns.append(ts.reward[0])

        experience, _ = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if False and step % eval_interval == 0:
            eval_op.run()
            avg_return = metric.result().numpy()
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    import IPython
    import base64

    video = open(filename,'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)

def create_policy_eval_video(py_env, tf_env, policy, filename, num_episodes=5, fps=30):
    import imageio

    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            action_step = None
            time_step = tf_env.reset()
            video.append_data(py_env.render())
            limit = 0
            while not time_step.is_last() and limit < 2000:
                action_step = policy.action(time_step, q_net.get_initial_state() if not action_step else action_step.state)
                time_step = tf_env.step(action_step.action)
                video.append_data(py_env.render())
                limit += 1
    return embed_mp4(filename)

create_policy_eval_video(eval_py_env, eval_env, eval_policy, "breakout-agent-trained")