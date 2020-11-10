from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_breast_cancer
import numpy as np
cancer = load_breast_cancer()

print(cancer.keys())
print(cancer['DESCR'])

print(cancer['feature_names'])

df_cancer = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
print(df_cancer.head())

df_target = pd.DataFrame(cancer['target'], columns=['cancer'])

# separando os daods
x_train, x_test, y_train, y_test = train_test_split(
    df_cancer, np.ravel(df_target), test_size=0.3, random_state=101)


# treinando os dados

model = SVC()

model.fit(x_train, y_train)

# avaliando modelo

pred = model.predict(x_test)
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


# usando grid seach

param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [
    1, 0.1, 0.01, 0.0001], 'kernel': ['rbf']}


grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)

grid.fit(x_train, y_train)


pred2 = grid.predict(x_test)

print(classification_report(y_test, pred2))

print(confusion_matrix(y_test, pred2))
