import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import tensorflow as tf
import pandas as pd


df = pd.read_csv('house_prices.csv')
print(df.head())

# pegando preco de x e y
x = df.iloc[:, 5:6].values
y = df.iloc[:, 2:3].values


# alterando escala
scaler_x = StandardScaler()
x_transformado = scaler_x.fit_transform(x)
print(x)

scaler_y = StandardScaler()
y_transformado = scaler_y.fit_transform(y)
print(y)


colunas = [tf.feature_column.numeric_column('x', shape=[1])]
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)

# separando os dados
x_train, x_test, y_train, y_test = train_test_split(
    x_transformado, y_transformado, test_size=0.3)


funcao_treinamento = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train}, y_train, batch_size=32, num_epochs=None, shuffle=True)

funcao_teste = tf.estimator.inputs.numpy_input_fn(
    {'x': x_test}, y_test, batch_size=32, num_epochs=1000, shuffle=False)

regressor.train(input_fn=funcao_treinamento, steps=10000)

metricas_trinamento = regressor.evaluate(
    input_fn=funcao_treinamento, steps=10000)

metricas_teste = regressor.evaluate(
    input_fn=funcao_teste, steps=10000)

print(metricas_trinamento)
print(metricas_teste)

novas_casas = np.array([[800], [900], [1000]])

novas_casas = scaler_x.transform(novas_casas)

print(novas_casas)

funcao_previsao = tf.estimator.inputs.numpy_input_fn(
    {'x': novas_casas}, shuffle=False)

previsoes = regressor.predict(input_fn=funcao_previsao)
print('precos previstos')
for p in regressor.predict(input_fn=funcao_previsao):
    print(scaler_y.inverse_transform(p['predictions']))
