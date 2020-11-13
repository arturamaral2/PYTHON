import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
a = tf.constant([[2, 2], [2, 2]], name='a')
b = tf.constant([[1, 2], [1, 2]], name='b'),
mult = tf.matmul(a, b)

with tf.Session() as sess:
    print(sess.run(mult))
