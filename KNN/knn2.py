from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("KNN_Project_Data")
print(df.head())


sns.pairplot(df, hue='TARGET CLASS', palette='coolwarm')
plt.show()

# padronizando os dados com standScaler, precisa padrozinar no metodo KNN

scaler = StandardScaler()


scaler.fit(df.drop('TARGET CLASS', axis=1))


DadosTransformados = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_param = pd.DataFrame(DadosTransformados, columns=df.columns[:-1])
print(df_param.head())

# treinando o algoritmo agora com os dados padronizados

x_train, x_test, y_train, y_test = train_test_split(
    df_param, df['TARGET CLASS'], test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)


# verificando o modelo

print('matrizes com k = 1 \n')
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))


# Escolhendo o valor de K

erro_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    erro_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(14, 8))

plt.plot(range(1, 40), erro_rate, color='blue',
         linestyle='dashed', marker='o', markerfacecolor="red")
plt.xlabel('K')
plt.ylabel('taxa de erro')

plt.show()

# TREINANDO MODELO PARA K = 30 , COMO VISTO NO GRAFICO
print('\n')
print('matrizes com K = 30 \n')
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(x_train, y_train)

pred = knn.predict(x_test)


# verificando o modelo

print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
