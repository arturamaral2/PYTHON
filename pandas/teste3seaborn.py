import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np

tips = sns.load_dataset('tips')

print(tips.head())

sns.barplot(x = 'sex', y = 'total_bill',data = tips)
plt.show()
x = np.linspace(0,5,11)
y = x*x 
plt.plot(x,y)
plt.show()


