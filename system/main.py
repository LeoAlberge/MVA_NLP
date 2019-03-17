from trees import ListTrees, leaves_labels, remove_leaves
from ovo import OvO
from PCGF import PCFGParser, PLexicon
from evaluation import accuracy
import numpy as np
import argparse


parser = argparse.ArgumentParser("simple_example")
parser.add_argument("sentence", help="A sentence to parse", type=str)
args = parser.parse_args()
print(args.counter + 1)


def main():
    ovo = OvO('data/polyglot.pkl', 'data/train.txt')
    train_trees = ListTrees('data/train.txt')

    par = PCFGParser()
    par.count_rules(train_trees.rules)

    print(np.array(par.non_terminal_rules_counts.values()))

    lex = PLexicon()
    lex.learn(train_trees.rules_unchomsky)

    print(ovo.most_similar_levenshtein('voile', k=5))

    print(ovo.most_similar('voile', True))
    print(lex.count_rules[ovo.most_similar('voile', verbose=True)])

    y_true = []
    y_pred = []
    test_trees = ListTrees('data/test.txt')
    lines = []
    for (c, tree) in enumerate(test_trees):
        print('{0:0.2f}% done'.format(100 * c / len(test_trees)))

        tree.un_chomsky_normal_form(unaryChar='&')
        test_sentence = ovo.modify_tokens(list(tree.flatten()), False)
        condition, parsed_tree = par.cky(test_sentence)
        if condition:
            parsed_tree.pretty_print()
            lines.append('( ' + ' '.join(str(parsed_tree).split()) + ')')
            y_true.extend(leaves_labels(tree))
            y_pred.extend(leaves_labels(parsed_tree))
            print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))
        print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))

        with open(filepath, 'w') as file:
            file.writelines("%s\n" % l for l in lines)
    print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))



if __name__ == '__main__':
        main()
