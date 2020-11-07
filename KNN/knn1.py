from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("ClassifiedData", index_col=1)

print(df.head())

scaler = StandardScaler()

scaler.fit(df.drop('TARGET CLASS', axis=1))

df_normalizado = scaler.transform(df.drop('TARGET CLASS', axis=1))
print(df_normalizado)

df_param = pd.DataFrame(df_normalizado, columns=df.columns[:-1])
print(df_param.head())
print(df.columns)
print(df_param.columns)
print(len(df.columns))

x_train, x_test, y_train, y_test = train_test_split(
    df_param, df['TARGET CLASS'], test_size=0.3)


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)

print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# calculo de cutuvelo
erro_rate = []
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred = knn.predict(x_test)
    erro_rate.append(np.mean(pred != y_test))

plt.figure(figsize=(14, 8))

plt.plot(range(1, 40), erro_rate, color="blue", linestyle="dashed", marker="o")

plt.show()
