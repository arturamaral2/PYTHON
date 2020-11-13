import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

vetor = tf.constant([5, 10, 15], name='vetor')

print(vetor)

soma = tf.Variable(vetor + 5, name='soma')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    s = print(sess.run(soma))

print(s)

valor = tf.Variable(0, name='valor')
init2 = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init2)
    for i in range(5):
        valor = valor + 1
        print(sess.run(valor))
