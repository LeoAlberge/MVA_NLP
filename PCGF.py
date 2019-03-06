from collections import defaultdict
from nltk.tree import Tree
import numpy as np


class PCFGParser(object):
    def __init__(self):
        self.lexicon_rules_counts = defaultdict(int)  # count non terminal symbols
        self.lexicon_counts = defaultdict(int)  # count rules A -> w
        self.non_terminal_counts = defaultdict(int)  # count rules A -> w
        self.non_terminal_rules_counts = defaultdict(int)  # count rules A -> B C

        self.probabilistic_lexicon = defaultdict(float)
        self.probabilistic_rules = defaultdict(float)

    def count_rules(self, rules):
        for rule in rules:
            if rule.is_lexical():
                self.lexicon_counts[rule.lhs()] += 1
                self.lexicon_rules_counts[rule.lhs(), rule.rhs()] += 1
            else:
                self.non_terminal_counts[rule.lhs()] += 1
                self.non_terminal_rules_counts[rule.lhs(), rule.rhs()] += 1

    def proba_lexicon(self, key, word):
        return float(self.lexicon_rules_counts[key, word])/self.lexicon_counts[key]

    def proba_rule(self, key, w1, w2):
        return float(self.non_terminal_rules_counts[key, (w1, w2)])/self.non_terminal_counts[key]

    def log_proba_lexicon(self, key, word):
        return np.log(float(self.lexicon_rules_counts[key, word])/self.lexicon_counts[key])

    def log_proba_rule(self, key, w1, w2):
        return np.log(float(self.non_terminal_rules_counts[key, (w1, w2)])/self.non_terminal_counts[key])

    def cky(self, sentence):
        pr = {}
        back = defaultdict()
        lexicon_rules = list(self.lexicon_counts.keys())
        non_terminal_rules = list(self.non_terminal_rules_counts.keys())
        for j in range(1, len(sentence)+1):
            word = tuple([sentence[j-1]])
            for key in lexicon_rules:
                if self.proba_lexicon(key, word) != 0:
                    pr[j-1, j, key] = self.log_proba_lexicon(key, word)

            for i in range(j-2, -1, -1):
                for k in range(i+1, j):
                    for A, (B, C) in non_terminal_rules:
                        if ((i, k, B) in pr) and ((k, j, C) in pr) and (pr[i, k, B] > -np.inf) and (pr[k, j, C] > -np.inf):
                            if self.proba_rule(A, B, C) != 0:
                                if (i, j, A) in pr and pr[i, j, A] < self.log_proba_rule(A, B, C) + pr[i, k, B] + pr[k, j, C]:
                                    pr[i, j, A] = self.log_proba_rule(A, B, C) + pr[i, k, B] + pr[k, j, C]
                                    back[i, j, A] = (k, B, C)
                                elif (i, j, A) not in pr:
                                    pr[i, j, A] = self.log_proba_rule(A, B, C) + pr[i, k, B] + pr[k, j, C]
                                    back[i, j, A] = (k, B, C)

            max_proba = -np.inf
            for A, (_, _) in non_terminal_rules:
                if (0, len(sentence), A) in pr:
                    if pr[0, len(sentence), A] > max_proba:
                        arg_max = A
        res = self.build_tree(sentence, back, 0, len(sentence), arg_max)
        res.un_chomsky_normal_form()
        return res

    def build_tree(self, sentence, back, i, j, node):
        tree = Tree(node._symbol, children=[])
        if (i, j) == (j-1, j):
            tree.append(sentence[j-1])
            return tree
        else:
            if (i, j, node) in back.keys():
                k, b, c = back[i, j, node]
                tree.append(self.build_tree(sentence, back, i, k, b))
                tree.append(self.build_tree(sentence, back, k, j, c))
                return tree
            else:
                return tree
