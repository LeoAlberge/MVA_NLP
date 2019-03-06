import re
from nltk.tree import Tree
import numpy as np
from collections import defaultdict

def read_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return lines


class CustomTree(Tree):
    def __init__(self, node, children=None):
        super(CustomTree, self).__init__(node, children)
        self.clean_labels(self)

    def clean_labels(self, tree):
        """

        :param tree:
        :return:
        """
        if tree.label().find('-'):
            tree.set_label(tree.label().split('-')[0])
        for child in tree:
            if isinstance(child, Tree):
                self.clean_labels(child)
            else:
                return 0


    def leaves_with_labels(self, label=None):
        """
        Return list with leaves token and label(POS)
        :param label:
        :return:
        """
        leaves = []
        for child in self:
            if isinstance(child, Tree):
                leaves.extend(child.leaves_with_labels(child.label()))
            else:
                leaves.append((child, label))
        return leaves

    def get_rules(self, tree, previous=None):
        """

        :param tree:
        :param previous:
        :return:
        """
        rules = []

        if len(tree) == 1:
            for child in tree:
                if isinstance(child, Tree) and self.is_terminal(child):
                    rules.append((tree.label(), child.label(), True))
        else:
            rule = []

            for child in tree:
                rule.append(child.label())
                if self.is_terminal(child):
                    rules.append((child.label(), child.label(), True))
                else:
                    rules.extend(self.get_rules(child, child.label()))

            rules.append((previous, rule, False))
        return rules

    @staticmethod
    def is_terminal(tree):
        """

        :param tree:
        :return:
        """
        if len(tree) == 1:
            for child in tree:
                if isinstance(child, Tree):
                    return False
                else:
                    return True
        else:
            return False


class ListTrees(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.trees = []
        self.lines = read_file(filepath)
        self.rules = []
        for line in self.lines:
            tree = CustomTree.fromstring(line)[0]
            tree.collapse_unary(True, True)
            tree.chomsky_normal_form()

            self.trees.append(tree)
            self.rules.extend(tree.productions())

    def __getitem__(self, item):
        return self.trees[item]

    def get_vocabulary(self):
        res = []
        for tree in self.trees:
            res.extend(tree.leaves())
        return list(set(res))

class PCFGParser(object):
    def __init__(self):
        self.lexicon_rules_counts = defaultdict(int)  # count non terminal symbols
        self.lexicon_counts = defaultdict(int)  # count rules A -> w
        self.non_terminal_counts = defaultdict(int) # count rules A -> w
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

    def CKY(self, sentence):
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
            for A, (B, C) in non_terminal_rules:
                if (0, len(sentence), A) in pr:
                    if pr[0, len(sentence), A] > max_proba:
                        arg_max = A

        return self.build_tree(sentence, back, 0, len(sentence), arg_max)

    def build_tree(self, sentence, back, i, j, node):
        tree = Tree(node._symbol, children=[])
        if (i,j) == (j-1, j):
            tree.append(sentence[j-1])
            return tree
        else:
            if (i,j,node) in back.keys():
                k, B, C = back[i, j, node]
                tree.append(self.build_tree(sentence, back, i, k, B))
                tree.append(self.build_tree(sentence, back, k, j, C))
                return tree
            else:
                return tree


#
# trees = ListTrees('data/sequoia-corpus.txt')
#
# par = PCFGParser()
# par.count_rules(trees.rules)
#
# parsetree = trees[1]
# parsetree.un_chomsky_normal_form()
# parsetree.pretty_print()
# print('cou')
# tree = par.CKY(parsetree.flatten())
# tree.un_chomsky_normal_form()
# tree.pretty_print()
# #print(par.non_terminal_rules_counts)

