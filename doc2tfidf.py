    
import numpy as np

def doc2onehot_tfidf(self, data):
    onehot = np.zeros((len(data), len(self.__voc)))
    tf = np.zeros((len(data), len(self.__voc)))
    for d, sentence in enumerate(data):
        for word in sentence:
            if word in self.__voc:
                pos = self.__voc.index(word)
                onehot[d][pos] = 1
                tf[d][pos] += 1
    row_sum = tf.sum(axis=1)+1
    tf = tf / row_sum[:, np.newaxis]
    ndw = onehot.sum(axis=0)
    idf = list(map(lambda x: np.log10(self.len_data) / (x+1), ndw))
    tfidf = tf * np.array(idf)
    res = np.asarray(tfidf, dtype='f')
    return res
