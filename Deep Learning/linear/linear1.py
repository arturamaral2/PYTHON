from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

x = np.array([[18], [23], [28], [33], [38], [43], [48], [53], [58], [63]])

print(x)

y = np.array([[871], [1132], [1042], [1356], [1488],
              [1638], [1569], [1754], [1866], [1900]])
print(y)

plt.scatter(x, y)
plt.show()

regressor = LinearRegression()


regressor.fit(x, y)

print(regressor.intercept_)
print(regressor.coef_)

previsoes = regressor.predict(x)

print(previsoes)

mean_erro = mean_absolute_error(y, previsoes)

mean_erro_square = mean_squared_error(y, previsoes)

print(mean_erro)
print(mean_erro_square)

plt.plot(x, y, 'o')
plt.plot(x, previsoes, color='red')
plt.show()
