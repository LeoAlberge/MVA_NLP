import re
from nltk.tree import Tree

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
                self.lexicon_counts[rule.lhs()]
                if rule.lhs() not in self.lexicon_counts:
                    self.lexicon_counts[rule.lhs()] = 1
                    self.lexicon_rules_counts[rule.lhs()] = {}
               else:
                    self.lexicon_counts[rule.lhs()] += 1

               if rule.rhs() not in self.lexicon_rules_counts[rule.lhs()]:
                    self.lexicon_rules_counts[rule.lhs()][rule.rhs()] = 1
               else:
                    self.lexicon_rules_counts[rule.lhs()] [rule.rhs()] += 1
            else:
                if rule.lhs() not in self.non_terminal_counts:
                    self.non_terminal_counts[rule.lhs()] = 1
                    self.non_terminal_rules_counts[rule.lhs()] = {}
                else:
                    self.non_terminal_counts[rule.lhs()] += 1

                if rule.rhs() not in self.non_terminal_rules_counts[rule.lhs()]:
                    self.non_terminal_rules_counts[rule.lhs()][rule.rhs()] = 1
                else:
                    self.non_terminal_rules_counts[rule.lhs()][rule.rhs()] += 1

    def proba_lexicon(self, key, word):
        return float(self.lexicon_rules_counts[key][word])/self.lexicon_counts[key]

    def proba_rule(self, key, w1, w2):
        return float(self.non_terminal_rules_counts[key][w1, w2]) / self.non_terminal_counts[key]

    def learn(self):
        for key in self.lexicon_counts.keys():
            self.probabilistic_lexicon[key] = {}
            for word in self.lexicon_rules_counts[key].keys():
                self.probabilistic_lexicon[key][word] = self.proba_lexicon(key, word)

        for key in self.non_terminal_counts.keys():
            self.probabilistic_rules[key] = {}
            for w1, w2 in self.non_terminal_rules_counts[key].keys():
                self.non_terminal_rules_counts[key][w1, w2] = self.proba_rule(key, w1, w2)

    def CKY(self, sentence):
        pr = defaultdict(float)
        for j in range(len(sentence)):
            word = sentence[j]



lines = read_file('data/sequoia-corpus.txt')
parsetree = CustomTree.fromstring(lines[1])[0]
parsetree.collapse_unary(True)
parsetree.chomsky_normal_form()
parsetree.pretty_print()

rules =parsetree.productions()
#print(parsetree.get_rules(parsetree[0], parsetree[0].label()))


par = PCFGParser()
par.count_rules(rules)
par.learn()
#print(par.non_terminal_rules_counts)