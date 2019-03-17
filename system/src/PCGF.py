from collections import defaultdict
from nltk.tree import Tree
import numpy as np
from timer import timeit


class PLexicon(object):
    def __init__(self):
        self.count_rules = {}
        self.count_words = defaultdict(int)

    def learn(self, counts):
        for count in counts:
            pos, key = str(count.lhs()), count.rhs()[0]
            self.count_words[key] += 1
            if key not in self.count_rules:
                self.count_rules[key] = defaultdict(int)
                self.count_rules[key][pos] = 1
            else:
                self.count_rules[key][pos] += 1

    def proba(self, word, pos):
        return self.count_rules[word][pos] / self.count_words[word]

    def predict(self, word):
        argmax = None
        max_proba = -np.inf
        for pos in self.count_rules[word]:
            if self.proba(word, pos) > max_proba:
                max_proba = self.proba(word, pos)
                argmax = pos
        return argmax

    def predict_sentence(self, sent):
        res = []
        for word in sent:
            res.append(self.predict(word))
        return res


class PCFGParser(object):
    def __init__(self):
        self.lexicon_rules_counts = defaultdict(int)  # count non terminal symbols
        self.lexicon_counts = defaultdict(int)  # count rules A -> w
        self.non_terminal_counts = defaultdict(int)  # count rules A -> w
        self.non_terminal_rules_counts = defaultdict(int)  # count rules A -> B C

        self.log_probabilistic_lexicon = defaultdict(float)
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
        res = np.log(float(self.non_terminal_rules_counts[key, (w1, w2)])/self.non_terminal_counts[key])
        return res

    @timeit
    def cky(self, sentence, remove_rules=True):
        """
        Main function of our probabilistic context-free grammar parser.
        :param sentence: sentence ot be parsed
        :param remove_rules:
        :return: condition: True if parsable, else False.
                 parsed_tree
        """
        log_pr = {}
        back = defaultdict()
        lexicon_rules = self.lexicon_counts.keys()
        if remove_rules:
            non_terminal_rules = [rule for rule, count in self.non_terminal_rules_counts.items() if count > 1]
        else:
            non_terminal_rules = self.non_terminal_rules_counts.keys()

        if len(sentence) > 1:
            for j in range(1, len(sentence)+1):
                word = tuple([sentence[j-1]])
                for key in lexicon_rules:
                    pr_lex = self.proba_lexicon(key, word)
                    if pr_lex != 0:
                        log_pr[j-1, j, key] = np.log(pr_lex)
                for i in range(j-2, -1, -1):
                    for k in range(i+1, j):
                        for A, (B, C) in non_terminal_rules:
                            if ((i, k, B) in log_pr) and ((k, j, C) in log_pr):
                                log_pr_ikb = log_pr[i, k, B]
                                log_pr_kjc = log_pr[k, j, C]
                                log_pr_abc = self.log_proba_rule(A, B, C)

                                if (i, j, A) not in log_pr:
                                    log_pr[i, j, A] = log_pr_abc + log_pr_ikb + log_pr_kjc
                                    back[i, j, A] = (k, B, C)
                                elif (i, j, A) in log_pr and log_pr[i, j, A] < log_pr_abc + log_pr_ikb + log_pr_kjc:
                                    log_pr[i, j, A] = log_pr_abc + log_pr_ikb + log_pr_kjc
                                    back[i, j, A] = (k, B, C)

            parsable = False
            h_prob = -np.inf
            for n1, n2, A in log_pr.keys():
                if (n1, n2, A) == (0, len(sentence), A):
                    if log_pr[0, len(sentence), A] > h_prob:
                        arg_max = A
                        parsable = True

            if not parsable:
                return False, None

            res = self.build_tree(sentence, back, 0, len(sentence), arg_max)
            res.set_label('SENT')
            res.un_chomsky_normal_form(unaryChar='&')

            return True, res

        else:
            word = tuple([sentence[0]])
            for key in lexicon_rules:
                if self.proba_lexicon(key, word) != 0:
                    log_pr[0, 1, key] = self.log_proba_lexicon(key, word)

            h_prob = -np.inf
            parsable = False
            for A in lexicon_rules:
                if (0, len(sentence), A) in log_pr:
                    if log_pr[0, len(sentence), A] > h_prob:
                        arg_max = A
                        parsable = True

            if not parsable:
                return False, None

            res = self.build_tree(sentence, back, 0, len(sentence), arg_max)
            res.set_label('SENT')
            res.un_chomsky_normal_form(unaryChar='&')

            return True, res

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
