from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

df = pd.read_csv('house_prices.csv')
print(df.head())
x = df.iloc[:, 5].values

print('\n')
x = x.reshape(-1, 1)
print(x.shape)
print(x)
y = df.iloc[:, 2:3].values
print(y.shape)


scaler_x = StandardScaler()
x_transformado = scaler_x.fit_transform(x)
print(x)

scaler_y = StandardScaler()
y_transformado = scaler_y.fit_transform(y)
print(y)

# plotando o grafico metros quadrados vs preco
plt.scatter(x_transformado, y_transformado)
plt.show()


# valores aleatorios para comecar a iteração
np.random.seed(0)
np.random.rand(2)


b0 = tf.Variable(0.54)
b1 = tf.Variable(0.71)

erro = tf.losses.mean_squared_error(y_transformado, (b0 + b1*x_transformado))
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


previsoes_escala_alterada = b0_final + b1_final*x_transformado

plt.plot(x_transformado, previsoes_escala_alterada, color='red')
plt.plot(x_transformado, y_transformado, 'o')

plt.show()

y1 = scaler_y.inverse_transform(y_transformado)

previsao_normal = scaler_y.inverse_transform(
    b0_final + b1_final * x_transformado)


erro_absolueto = mean_absolute_error(y1, previsao_normal)
erro_quadrado = mean_squared_error(y1, previsao_normal)

print(erro_absolueto)
print(erro_quadrado)
