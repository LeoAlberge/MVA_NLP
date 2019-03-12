import pickle
import numpy as np
import io
from levenshtein_distance import levenshtein

from trees import ListTrees

with io.open('data/polyglot.pkl', 'rb') as f:
    words, embeddings = pickle.load(f, encoding='latin-1')


class OvO(object):
    def __init__(self, embeddings_filepath, train_filepath):
        with io.open(embeddings_filepath, 'rb') as file:
            words, embeddings = pickle.load(file, encoding='latin-1')

        self.vocabulary = ListTrees(filepath=train_filepath).get_vocabulary()
        self.id2vocabulary = {w: id for w, id in enumerate(self.vocabulary)}
        self.vocabulary2id = {v: k for k, v in self.id2vocabulary.items()}
        self.vocabulary = np.array(self.vocabulary).reshape(-1, 1)

        self.words = words
        self.embeddings = np.array(embeddings)

        self.id2word = {w: id for w, id in enumerate(self.words)}
        self.word2id = {v: k for k, v in self.id2word.items()}
        self.words = np.array(words).reshape(-1, 1)

    def most_similar_embeddings(self, w, k=5):
        # K most similar words
        # We exclude the first word because it's the words itself
        if w in self.word2id:
            k_ids = np.apply_along_axis(arr=self.vocabulary,
                                        func1d=lambda x: -self.embedding_score(x[0], w),
                                        axis=1).argsort()[0:k]

            return [self.id2vocabulary[word] for word in k_ids]
        else:
            return None

    def most_similar_levenshtein(self, w, k=5):
        # K most similar words
        k_ids = np.apply_along_axis(arr=self.vocabulary,
                                    func1d=lambda x: levenshtein(x[0], w),
                                    axis=1).argsort()[0:k]
        return [self.id2vocabulary[word] for word in k_ids]

    def most_similar_v2_func(self, w, k=5):
        # K most similar words
        k_ids = np.apply_along_axis(arr=self.vocabulary,
                                    func1d=lambda x: self.embedding_score(x[0], w)*(1-levenshtein(x[0], w)/max(len(x[0]), len(w))),
                                    axis=1).argsort()[:-k]
        return [self.id2vocabulary[word] for word in k_ids]

    def embedding_score(self, w1, w2):
        # cosine similarity: np.dot  -  np.linalg.norm
        # when error (key error etc) return None
        try:
            w1_id = self.word2id[w1]
            w2_id = self.word2id[w2]
            return self.embeddings[w1_id].dot(self.embeddings[w2_id])/(np.linalg.norm(self.embeddings[w2_id])*np.linalg.norm(self.embeddings[w1_id]))
        except KeyError:
            return 0

    def most_similar_v2(self, w, verbose):
        if w in self.vocabulary2id.keys():
            return w
        elif w in self.word2id.keys():
            if verbose:
                print('{} in word2id: most similar: {}'.format(w, self.most_similar_v2_func(w, 1)[0]))
            return self.most_similar_v2_func(w, 1)[0]
        else:
            if verbose:
                print(
                    '{} not in word2id: Levenshtein most similar: {}'.format(w, self.most_similar_levenshtein(w, 1)[0]))
            return self.most_similar_levenshtein(w, 1)[0]

    def most_similar(self, w, verbose):
        if w in self.vocabulary2id.keys():
            return w
        elif w in self.word2id.keys():
            if verbose:
                print('{} in word2id: most similar: {}'.format(w, self.most_similar_embeddings(w, 1)[0]))
            return self.most_similar_embeddings(w, 1)[0]
        else:
            if verbose:
                print('{} not in word2id: Levenshtein most similar: {}'.format(w, self.most_similar_levenshtein(w, 1)[0]))
            return self.most_similar_levenshtein(w, 1)[0]

    def modify_tokens(self, tokens, verbose=False):
        res = []
        for token in tokens:
            res.append(self.most_similar(token, verbose))

        assert(len(tokens) == len(res))
        return res


