import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


valor1 = tf.constant(10)

soma = tf.Variable(5 + valor1, name='valor1')

print(soma)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    s = sess.run(soma)


print(f" o valor é {s}")
