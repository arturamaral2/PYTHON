import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

valor1 = tf.constant(1)
valor2 = tf.constant(2)

print(valor1)

soma = valor1 + valor2

print(soma)

with tf.Session() as sess:
    s = sess.run(soma)

print(s)

texto1 = tf.constant('texto 1')
texto2 = tf.constant('texto 2')

print(texto1)

with tf.Session() as sess:
    con = sess.run(texto1 + texto2)

print(con)
