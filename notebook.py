import tensorflow as tf

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory

data_spec = tf.TensorSpec((), dtype=tf.int32, name='state')

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=data_spec,
    batch_size=1,
    max_length=10
)

replay_buffer.add_batch(tf.constant(1, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(2, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(3, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(4, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(5, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(6, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(7, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(8, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(9, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(10, dtype=tf.int32))
replay_buffer.add_batch(tf.constant(11, dtype=tf.int32))

x = replay_buffer.get_next(num_steps=2)
x = replay_buffer.get_next(num_steps=2)
x = replay_buffer.get_next(num_steps=2)

pass

