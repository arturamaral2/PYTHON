import pandas as pd 
import numpy as np 

df = pd.DataFrame(np.random.rand(5,4), index = 'A B C D E'.split() ,columns = 'X W Y Z '.split())
print(df)
print(df.loc['A'])
print(df[df['W']>0.5]['Y'])