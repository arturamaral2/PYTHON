from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


x = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])

y = np.array([[871], [1132], [1042], [1356], [1488],
              [1638], [1569], [1754], [1866], [1900]])


scaler_x = StandardScaler()
x = scaler_x.fit_transform(x)
print(x)

scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)
print(y)

plt.plot(x, y)
plt.show()

np.random.seed(0)
np.random.rand(2)


b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)
erro = tf.losses.mean_squared_error(y, (b0 + b1*x))

otimizador = tf.train.GradientDescentOptimizer(learning_rate=0.001)

treinamento = otimizador.minimize(erro)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(treinamento)

    b0_final, b1_final = sess.run([b0, b1])

print(b0_final)
print(b1_final)

# as previsoes em escala mudada
previsoes_escala_alterada = b0_final + b1_final*x

plt.plot(x, previsoes_escala_alterada, color='red')
plt.plot(x, y, 'o')

plt.show()

# presivoes em escala normal
previsao_normal = scaler_y.inverse_transform(
    b0_final + b1_final * x)

# voltando o y para escala normal
y1 = scaler_y.inverse_transform(y)

print(y1)
print(previsao_normal)


erro_absolueto = mean_absolute_error(y1, previsao_normal)
erro_quadrado = mean_squared_error(y1, previsao_normal)

print(erro_absolueto)
print(erro_quadrado)
