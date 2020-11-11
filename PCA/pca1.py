import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.datasets import load_breast_cancer


cancer = load_breast_cancer()

type

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df.head())


scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)

pca = PCA(n_components=2)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

print(x_pca.shape)
print(scaled_data.shape)

# plotando componente
plt.figure(figsize=(12, 7))
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('Primeiro Componente principal')
plt.ylabel('Segundo componente Principal')
plt.show()

df_comp = pd.DataFrame(pca.components_, columns=cancer['feature_names'])
plt.figure(figsize=(12, 7))
sns.heatmap(df_comp, cmap='plasma')
plt.show()
print(df_comp.head())
