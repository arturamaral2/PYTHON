from sklearn.cluster import KMeans
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.8, random_state=50)

plt.figure(figsize=(12, 8))
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
plt.show()


kmeans = KMeans(n_clusters=4)

kmeans.fit(data[0])


f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 8))
ax1.set_title('Original')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')
ax2.set_title('Modelo')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')
plt.show()
