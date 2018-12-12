    
import numpy as np

def __bow(data, freq):
    fd = {}
    for mail in data:
        for token in mail:
            if token in fd:
                fd[token] += 1
            else:
                fd[token] = 1
    fd = sorted(fd.items(),
                key=lambda items: items[1],
                reverse=True)
    l1 = [w for (w, f) in list(fd)[:int(len(list(fd)) / freq)]]
    l2 = [w for (w, f) in list(fd) if f > freq]
    return l1 if len(l1) < len(l2) else l2

def doc2onehot_tfidf(self, data):
    __voc = __bow(data, 10)
    onehot = np.zeros((len(data), len(__voc)))
    tf = np.zeros((len(data), len(__voc)))
    for d, sentence in enumerate(data):
        for word in sentence:
            if word in __voc:
                pos = __voc.index(word)
                onehot[d][pos] = 1
                tf[d][pos] += 1
    row_sum = tf.sum(axis=1)+1
    tf = tf / row_sum[:, np.newaxis]
    ndw = onehot.sum(axis=0)
    idf = list(map(lambda x: np.log10(self.len_data) / (x+1), ndw))
    tfidf = tf * np.array(idf)
    res = np.asarray(tfidf, dtype='f')
    return res
