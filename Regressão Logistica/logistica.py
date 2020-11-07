from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib.pyplot import axis
import numpy as np
from numpy.lib.twodim_base import tri
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

train = pd.read_csv('titanic_train.csv')

print(train.head())
print(train.info())
# visualizando os dados
sns.heatmap(train.isnull(), cmap='viridis')
plt.show()

sns.countplot(data=train, x='Survived', hue='Sex', palette='RdBu_r')
plt.show()

# tratando os dados faltantes de idade

sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()


def inputIdade(cols):
    idade = cols[0]
    classe = cols[1]
    if pd.isnull(idade):
        if classe == 1:
            return 37
        elif classe == 2:
            return 29
        else:
            return 24

    return idade


train['Age'] = train[['Age', 'Pclass']].apply(inputIdade, axis=1)

# DEletando dado cabine e dados nulos
train.drop('Cabin', inplace=True, axis=1)
train.dropna(inplace=True)

sns.heatmap(train.isnull(), cmap='viridis')
plt.show()


# colocando dados para serem usados com get_dummies
sex = pd.get_dummies(train['Sex'], drop_first=True)


embark = pd.get_dummies(train['Embarked'], drop_first=True)

train.drop(['Sex', 'PassengerId', 'Name', 'Ticket',
            'Embarked'], axis=1, inplace=True)

train = pd.concat([train, sex, embark], axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    train.drop('Survived', axis=1), train['Survived'], test_size=0.3)


logmodel = LogisticRegression()

logmodel.fit(x_train, y_train)

pred = logmodel.predict(x_test)


print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
