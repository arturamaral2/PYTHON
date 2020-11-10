from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


loan = pd.read_csv("loan_data.csv")

print(loan.info())
print(loan.head())
print(loan.describe())

# analise exploratioria dos dados
plt.figure(figsize=(12, 7))
loan[loan['credit.policy'] == 1]['fico'].hist(
    alpha=0.5, color='blue', bins=30, label='Credity Policy = 1')
loan[loan['credit.policy'] == 0]['fico'].hist(
    alpha=0.5, color='red', bins=30, label='Credity Policy = 0')


plt.show()


plt.figure(figsize=(12, 7))
loan[loan['not.fully.paid'] == 1]['fico'].hist(
    alpha=0.5, color='blue', bins=30, label='not.fully.paid= 1')
loan[loan['not.fully.paid'] == 0]['fico'].hist(
    alpha=0.5, color='red', bins=30, label='not.fully.paid = 0')

plt.show()

plt.figure(figsize=(12, 7))
sns.countplot(x='purpose', hue='not.fully.paid', data=loan, palette='Set1')
plt.show()

sns.jointplot(x='fico', y='int.rate', data=loan, color='purple')
plt.show()


sns.lmplot(x='fico', y='int.rate', data=loan,
           hue='credit.policy', palette='Set1')

plt.show()


# tratando os dados

cat_feats = ['purpose']
final_data = pd.get_dummies(loan, columns=cat_feats, drop_first=True)


print(final_data.info())
print(final_data.head())

# treinando o modelo de arvore
x = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

# separando dados
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3)

dtree = DecisionTreeClassifier()

dtree.fit(x_train, y_train)
pred = dtree.predict(x_test)
tree.plot_tree(dtree)
plt.show()
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))

# treiando floresta
print('\n')
florest = RandomForestClassifier(n_estimators=500)

florest.fit(x_train, y_train)

pred2 = florest.predict(x_test)


print(classification_report(y_test, pred2))
print(confusion_matrix(y_test, pred2))
