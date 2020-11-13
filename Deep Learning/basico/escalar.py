import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


a = tf.constant([-1.0, 2.0, 3.0], name='entradas')
b = tf.constant([5.4, 3.3, 3.7], name='pesos')

mult = tf.multiply(a, b)
soma = tf.reduce_sum(mult)
with tf.Session() as sess:
    print(sess.run(a))
    print('\n')
    print(sess.run(b))
    print('\n')
    print(sess.run(mult))
    print('\n')
    print(sess.run(soma))
