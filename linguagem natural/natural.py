from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import nltk

# nltk.download() acessar download da bibliboteca


messagens = [line.rstrip() for line in open(
    'smsspamcollection/SMSSpamCollection')]


print(len(messagens))

for message_number, messagem in enumerate(messagens[:15]):
    print(message_number, messagem)

print('\n')


messagens2 = pd.read_csv(
    'smsspamcollection/SMSSpamCollection', sep='\t', names=['label', 'message'])

print(messagens2.describe())
print(messagens2.groupby('label').describe())

# analise exploratia de dados

messagens2['lenght'] = messagens2['message'].apply(len)
print(messagens2.head())

messagens2['lenght'].plot(kind='hist', bins=50)
plt.show()

messagens2.hist(bins=100, column='lenght', by='label')
plt.show()


mess = 'Mensagem exemplo " notem : possui pontuacao !!! '

sempont = [car for car in mess if car not in string.punctuation]

print(sempont)
sempont = ''.join(sempont)
print(sempont)
# palavra comuns em ingles que não significam nada
stopwords.words('english')


def text_process(mess):
    # retira as pontuaçoes
    nopunc = [char for char in mess if char not in string.punctuation]
    # juntando novamente
    nopunc = ''.join(nopunc)

    # removendo as stopword
    sms = [word for word in nopunc.split() if word.lower()
           not in stopwords.words('english')]

    return sms


print(messagens2['message'].head().apply(text_process))


# treinando modelo
bow_transformer = CountVectorizer(
    analyzer=text_process).fit(messagens2['message'])

# transofrmando no modelo
messages_bow = bow_transformer.transform(messagens2['message'])

print(messages_bow.shape)
print(messages_bow.nnz)


tdidf_transforme = TfidfTransformer()

tdidf_transforme = tdidf_transforme.fit(messages_bow)

# fitando modelo
message_tfidf = tdidf_transforme.transform(messages_bow)
spam_detect_model = MultinomialNB().fit(message_tfidf, messagens2['label'])
