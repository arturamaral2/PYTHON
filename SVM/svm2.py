from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
iris = sns.load_dataset('iris')

print(iris.head())


sns.pairplot(iris, hue='species', palette='Dark2')

plt.show()

setosa = iris[iris['species'] == 'setosa']
sns.kdeplot(x='sepal_width', y='sepal_length',
            data=setosa, cmap='plasma', shade=True, shade_lowest=False)

plt.show()


# divisão treino teste

x = iris.drop('species', inplace=False, axis=1)
y = iris['species']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)


model = SVC()

model.fit(x_train, y_train)

pred = model.predict(x_test)

# analisando modelo


print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
