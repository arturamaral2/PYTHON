import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

tf.reset_default_graph()

a = tf.add(2, 2, name='add')
b = tf.multiply(a, 3, name='mult1')
c = tf.multiply(b, a, name='mult2')

with tf.Session() as sess:
    writer = tf.summary.FileWriter('output', sess.graph)
    print(sess.run(c))
    writer.close()
