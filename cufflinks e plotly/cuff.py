import cufflinks as cf
import pandas as pd
import plotly.offline as py
import plotly.graph_objs as go
import numpy as np
py.init_notebook_mode(connected=True)
cf.go_offline()

df = pd.DataFrame(np.random.randn(100, 4), columns='A B C D'.split())
print(df.head())

df2 = pd.DataFrame({'Categoria': ['A', 'B', 'C'], 'Valores': [32, 43, 50]})

print(df2.head())

trace = go.Bar(x=['Banana', 'Maçã', 'Uva'],
               y=[10, 20, 30])
data = [trace]
py.iplot(data)
