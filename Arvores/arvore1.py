from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("kyphosis.csv", sep=',')

print(df)
print(df.info())

# avaliando os dados
sns.pairplot(df, hue="Kyphosis")
plt.show()

x = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']
# separando os dados
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# instanciando objetov da arvore de decisao
dtree = DecisionTreeClassifier()
# treinando
dtree2 = dtree.fit(x_train, y_train)
pred = dtree.predict(x_test)

# avaliando o modelo
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

print('\n')

# usando floresta


florest = RandomForestClassifier()
# treiando floresta
florest.fit(x_train, y_train)

pred_florest = florest.predict(x_test)
# avaliando a floresta

print(classification_report(y_test, pred_florest))
print(confusion_matrix(y_test, pred_florest))


tree.plot_tree(dtree)
plt.show()
