import numpy as np 

minha_lista = [1,2,3]
print(minha_lista)

print(np.array(minha_lista))

sequencia1 = np.arange(0,10)
print(sequencia1)
sequencia2 = np.random.rand(25)

sequencia2.resize((5,5))

print(sequencia2.shape)
print(sequencia2)
print(sequencia2.mean())
