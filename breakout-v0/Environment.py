import numpy as np
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

class BreakoutEnv(py_environment.PyEnvironment):
    def __init__(self, env: py_environment.PyEnvironment):
        self._env = env
        self._action_spec = self._env.action_spec()

        old_obs_spec = self._env.observation_spec() # type: array_spec.BoundedArraySpec
        self._observation_spec = array_spec.BoundedArraySpec(
                                                shape=(100, 100, 1),
                                                dtype=np.float32,
                                                name=old_obs_spec.name,
                                                minimum=old_obs_spec.minimum,
                                                maximum=old_obs_spec.maximum)

        self.reset()
    
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        initial_state = self._env.reset()
        initial_state = self._transform_state(initial_state.observation)
        # self._states = np.stack([initial_state]*4, axis=2)
        return ts.restart(initial_state)
    
    def _step(self, action):
        t_step = self._env.step(action)
        state = self._transform_state(t_step.observation)
        return ts.TimeStep(step_type=t_step.step_type, reward=t_step.reward, discount=t_step.discount, observation=state)


    def _transform_state(self, input):
        grey_scale = tf.image.rgb_to_grayscale(input)
        resized = tf.image.resize(grey_scale, [100, 100], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        output = tf.image.convert_image_dtype(resized, tf.float32)
        return output.numpy()

    def render(self):
        return self._env.render()
