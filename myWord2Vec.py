import seaborn as sb
import numpy as np
words = ['queen', 'book', 'king', 'magazine', 'car', 'bike']
vectors = np.array([[0.1,   0.3],  # queen
                    [-0.5, -0.1],  # book
                    [0.2,   0.2],  # king
                    [-0.3, -0.2],  # magazine
                    [-0.5,  0.4],  # car
                    [-0.45, 0.3]]) # bike


vectors.sum(axis = 0)
sb.plt.plot(vectors[:,0] , vectors[:,1] , 'o')


sentences = [
    'the king loves the queen',
    'the queen loves the king',
    'the dwarf hates the king',
    'the queen hates the dwarf',
    'the dwarf poisons the king',
    'the dwarf poisons the queen']


from collections import defaultdict

def Vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary

def docs2bow(docs, dictionary):
    """Transforms a list of strings into a list of lists where
    each unique item is converted into a unique integer."""
    for doc in docs:
        yield [dictionary[word] for word in doc.split()]

vocabulary = Vocabulary()
sentences_bow = list(docs2bow(sentences, vocabulary))



V, N = len(vocabulary), 3
WI = (np.random.random((V, N)) - 0.5) / N
WO = (np.random.random((N, V)) - 0.5) / N



import numpy as np
np.dot(WI[vocabulary['dwarf']] , WO.T[vocabulary['hates']])


p = (np.exp(-np.dot(WI[vocabulary['dwarf']],
                   WO.T[vocabulary['hates']])) /
     sum(np.exp(-np.dot(WI[vocabulary['dwarf']],
                       WO.T[vocabulary[w]]))
         for w in vocabulary))


tmp =     (np.exp(-np.dot(WI[vocabulary['dwarf']], WO.T[vocabulary[w]])) for w in vocabulary)


print(p)



target_word = 'king'
input_word = 'queen'
learning_rate = 1.0

for word in vocabulary:
    p_word_queen = (np.exp(
        -np.dot(WO.T[vocabulary[word]], WI[vocabulary[input_word]])
    ) /
        sum(np.exp(-np.dot(WO.T[vocabulary[w]], WI[vocabulary[input_word]]))
            for w in vocabulary)
    )

    t = 1 if word == target_word else 0
    error = t - p_word_queen
    WO.T[vocabulary[word]] = (WO.T[vocabulary[word]] - learning_rate * error * WI[vocabulary[input_word]])


print(WO)