import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

column_name = ['user_id', 'item_id', 'rating', 'timestamp']
# pegando dados
df = pd.read_csv("u.data", sep='\t', names=column_name)

print(df.head())

# pegando filmes
movie_title = pd.read_csv('Movie_id_Titles')


print(movie_title.head())

# juntando usarios com filmes
df = pd.merge(df, movie_title, on='item_id')
print('cabecalho filmes unidos \n')
print(df.head())

# analise de dados

print(df.groupby('title')['rating'].mean(
).sort_values(ascending=False).head(10))


print(df.groupby('title')['rating'].count(
).sort_values(ascending=False).head(10))

rating = pd.DataFrame(df.groupby('title')['rating'].mean())
print(rating.head())

rating['count'] = pd.DataFrame(df.groupby('title')['rating'].count())
print(rating.head())


plt.figure(figsize=(12, 7))
rating['count'].hist(bins=70)
plt.show()


plt.figure(figsize=(12, 7))
rating['rating'].hist(bins=70)
plt.show()

sns.jointplot(x='rating', y='count', data=rating, alpha=0.4, height=8)
plt.show()


movie_mat = df.pivot_table(index='user_id', columns='title', values='rating')
print(rating.sort_values('count', ascending=False).head())


star_wars_user_rating = movie_mat['Star Wars (1977)']
liarliar_user_rating = movie_mat['Liar Liar (1997)']
print(star_wars_user_rating.head())

# vendo filmes similares

similares_to_star_wars = movie_mat.corrwith(star_wars_user_rating)
similar_to_liar_liar = movie_mat.corrwith(liarliar_user_rating)

corr_starwars = pd.DataFrame(similares_to_star_wars, columns=['Correlation'])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
print(movie_mat.head())

# muitos filmes com corelaçao igual a 1, mas que não tem correlação na realidade, poucas pessoas viram os dois e avaliaram bem
print(corr_starwars.sort_values('Correlation', ascending=False).head(10))


corr_starwars = corr_starwars.join(rating['count'])
print(corr_starwars.head(10))


Df_final_star_Wars = corr_starwars[corr_starwars['count'] > 100].sort_values(
    'Correlation', ascending=False)

print(Df_final_star_Wars.head(10))

# fazendo a mesma coisa para liar liar

corr_liar_liar = pd.DataFrame(similar_to_liar_liar, columns=['Correlation'])
corr_liar_liar.dropna(inplace=True)

print(corr_liar_liar.head())

# muito filmes nada a ver, com correlação alta

print(corr_liar_liar.sort_values('Correlation', ascending=False).head(10))

corr_liar_liar = corr_liar_liar.join(rating['count'])

# correlação com a contagem de filmes
print(corr_liar_liar.head())

df_final_liar_liar = corr_liar_liar[corr_liar_liar['count'] > 100].sort_values(
    'Correlation', ascending=False)

print(df_final_liar_liar.head(10))
