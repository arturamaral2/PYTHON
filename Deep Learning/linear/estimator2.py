from sklearn.model_selection import train_test_split
import tensorflow as tf
from matplotlib.pyplot import axis
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
df = pd.read_csv('house_prices.csv')
print(df.head())

print(df.columns)
colunas_usadas = ['price', 'bedrooms', 'bathrooms', 'sqft_living',
                  'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                  'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                  'lat', 'long']

df = pd.read_csv('house_prices.csv', usecols=colunas_usadas)
print(df.head())

scaler_x = MinMaxScaler()

df[['bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
    'lat', 'long']] = scaler_x.fit_transform(df[['bedrooms', 'bathrooms', 'sqft_living',
                                                 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade',
                                                 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
                                                               'lat', 'long']])
print(df.head())

scaler_y = MinMaxScaler()

df[['price']] = scaler_y.fit_transform(df[['price']])

x = df.drop('price', axis=1)

y = df['price']

print(x.head())
print(type(y))

# excluindo o preco
previsores_colunas = colunas_usadas[1:]
print(previsores_colunas)
colunas = [tf.feature_column.numeric_column(key=c)for c in previsores_colunas]

print(colunas[0])
# separando os dados
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)

print(x_train.shape)

# definindo as funcao de treinamento e teste
funcao_treinamento = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, batch_size=32, num_epochs=None, shuffle=True)

funcao_teste = tf.estimator.inputs.pandas_input_fn(
    x=x_test, y=y_test, batch_size=32, num_epochs=10000, shuffle=False)


# trinando
regressor = tf.estimator.LinearRegressor(feature_columns=colunas)

regressor.train(funcao_treinamento)

metricas_treinamento = regressor.evaluate(
    input_fn=funcao_treinamento, steps=10000)

metricas_teste = regressor.evaluate(input_fn=funcao_teste, steps=10000)

print('metricas treinamento')
print(metricas_treinamento)
print('metricas teste')
print(metricas_teste)
