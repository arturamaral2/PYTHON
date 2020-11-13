import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

p = tf.placeholder('float', None)
operacao = p + 2.0

with tf.Session() as sess:
    resultado = sess.run(operacao, feed_dict={p: [1.0, 2.0, 3.0]})

    print(resultado)

p2 = tf.placeholder('float', [None, 5])

dados = [[5, 4, 4, 4, 4], [2, 1, 1, 1, 1]]
operacao2 = p2 * 5
with tf.Session() as sess:
    resultado2 = sess.run(operacao2, feed_dict={p2: dados})
    print(resultado2)
