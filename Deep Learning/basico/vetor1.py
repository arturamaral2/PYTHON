import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

a = tf.constant([1, 2, 3], name='a')
b = tf.constant([2, 3, 4], name='b')

soma = a + b

with tf.Session() as sess:
    print(sess.run(soma))

a1 = tf.constant([[1, 2, 3], [4, 5, 6]])

b1 = tf.constant([[1, 2, 3], [4, 5, 6]])

soma1 = tf.add(a1, b1)

with tf.Session() as sess:
    print(sess.run(a1))
    print('\n')
    print(sess.run(b1))
    print('\n')
    print(sess.run(soma1))


a2 = tf.constant([[1, 2, 3], [4, 5, 6]])

b2 = tf.constant([[1], [4]])
soma2 = a2+b2

with tf.Session() as sess:
    print(sess.run(a2))
    print('\n')
    print(sess.run(b2))
    print('\n')
    print(sess.run(soma2))
