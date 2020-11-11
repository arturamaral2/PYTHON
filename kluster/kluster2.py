from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('College_Data', index_col=0)


print(df.head())
print(df.info())

print(df.describe)
sns.set_style('whitegrid')
sns.lmplot(x='Grad.Rate', y='Room.Board', hue='Private',
           data=df, fit_reg=False, height=6, palette='coolwarm')
plt.show()


sns.set_style('whitegrid')
sns.lmplot(x='Outstate', y='F.Undergrad', hue='Private',
           data=df, fit_reg=False, height=12, palette='coolwarm')
plt.show()

sns.set_style('darkgrid')
g = sns.FacetGrid(df, hue='Private', height=6)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=0.5)
plt.show()

sns.set_style('darkgrid')
g = sns.FacetGrid(df, hue='Private', height=6, aspect=2)
g = g.map(plt.hist, 'Grad.Rate', bins=20, alpha=0.5)
plt.show()

print(df[df['Grad.Rate'] > 100])

df['Grad.Rate'][df['Grad.Rate'] > 100] = 100

print(df.loc['Cazenovia College'])


# criando modelo de kmeans


kmeans = KMeans(n_clusters=2)

kmeans.fit(df.drop('Private', inplace=False, axis=1))

print(kmeans.cluster_centers_)


def converter(cluster):
    if cluster == 'Yes':
        return 1
    else:
        return 0


df['Cluster'] = df['Private'].apply(converter)

print(df.head())


print(classification_report(df['Cluster'], kmeans.labels_))
print(confusion_matrix(df['Cluster'], kmeans.labels_))
