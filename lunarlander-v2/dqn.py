import tensorflow as tf

from tf_agents.environments import suite_gym, tf_py_environment, wrappers
from tf_agents.policies import random_py_policy

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
    import os

    filename = filename + ".mp4"
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            action_step = None
            time_step = tf_env.reset()
            video.append_data(py_env.render())
            step = 0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = tf_env.step(action_step.action)
                video.append_data(py_env.render())
                step += 1
                if step % 5 == 0:
                    print('Final Reward: ', time_step.reward)
            print('Final Reward: ', time_step.reward)
    return embed_mp4(filename)

env_name = 'LunarLander-v2'
train_py_env = suite_gym.load(env_name)
train_env = tf_py_environment.TFPyEnvironment(train_py_env)

print('Action Spec: ', train_py_env.action_spec())
print('Observation Sepc: ', train_py_env.observation_spec())
print('Time Spec: ', train_py_env.time_step_spec())

random_policy = random_py_policy.RandomPyPolicy(train_py_env.time_step_spec(), train_py_env.action_spec())
create_policy_eval_video(train_py_env, train_env, random_policy, r'videos\lunar_landing_random')