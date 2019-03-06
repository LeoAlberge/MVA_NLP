import pickle
import numpy as np
import codecs
import io
from levenshtein_distance import levenshtein

from parse_file import ListTrees

with io.open('data/polyglot.pkl', 'rb') as f:
    words, embeddings = pickle.load(f, encoding='latin-1')


class OvO(object):
    def __init__(self, embeddings_filepath, train_filepath):
        with io.open(embeddings_filepath, 'rb') as f:
            words, embeddings = pickle.load(f, encoding='latin-1')

        self.vocabulary = ListTrees(filepath=train_filepath).get_vocabulary()
        self.id2vocabulary = {w: id for w, id in enumerate(self.vocabulary)}
        self.vocabulary2id = {v: k for k, v in self.id2vocabulary.items()}
        self.vocabulary = np.array(self.vocabulary).reshape(-1, 1)

        self.words = words
        self.embeddings = np.array(embeddings)

        self.id2word = {w: id for w, id in enumerate(self.words)}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.words = np.array(words).reshape(-1,1)

    def most_similar_embeddings(self, w, K=5):
        # K most similar words
        # We exclude the first word because it's the words itself
        if w in self.word2id:
            K_ids = np.apply_along_axis(arr=self.vocabulary,
                                        func1d=lambda x: -self.embedding_score(x[0], w),
                                        axis=1).argsort()[0:K]

            return [self.id2vocabulary[word] for word in K_ids]
        else:
            return None

    def most_similar_levenshtein(self, w, K=5):
        # K most similar words
        K_ids = np.apply_along_axis(arr=self.vocabulary,
                                    func1d=lambda x: levenshtein(x[0], w),
                                    axis=1).argsort()[0:K]
        return [self.id2vocabulary[word] for word in K_ids]

    def embedding_score(self, w1, w2):
        # cosine similarity: np.dot  -  np.linalg.norm
        # when error (key error etc) return None
        try:
            w1_id = self.word2id[w1]
            w2_id = self.word2id[w2]
            return self.embeddings[w1_id].dot(self.embeddings[w2_id])/(np.linalg.norm(self.embeddings[w2_id])*np.linalg.norm(self.embeddings[w1_id]))
        except KeyError:
            return 0

    def most_similar(self, w):
        if w in self.vocabulary2id.keys():
            return w
        elif w in self.word2id.keys():
            return self.most_similar_embeddings(w, 1)[0]
        else:
            return self.most_similar_levenshtein(w, 1)[0]


o = OvO('data/polyglot.pkl', 'data/train.txt')


print(o.most_similar('chien'))


