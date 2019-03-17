import numpy as np
from trees import ListTrees, leaves_labels, remove_leaves
from ovo import OvO
from PCGF import PCFGParser, PLexicon

def accuracy(list_pos_predicted, list_pos_true):
    """

    :param list_pos_predicted:
    :param list_pos_true:
    :return: Accuracy
    """
    return (np.array(list_pos_predicted)==np.array(list_pos_true)).sum()/len(list_pos_predicted)


def evaluate_parser(filepath):
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
        print('{0:0.2f}% done'.format(100*c/len(test_trees)))

        tree.un_chomsky_normal_form(unaryChar='&')
        test_sentence = ovo.modify_tokens(list(tree.flatten()), False)
        condition, parsed_tree = par.cky(test_sentence)
        if condition:
            parsed_tree.pretty_print()
            lines.append('( '+' '.join(str(parsed_tree).split()) + ')')
            y_true.extend(leaves_labels(tree))
            y_pred.extend(leaves_labels(parsed_tree))
            print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))
        print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))

        with open(filepath, 'w') as file:
            file.writelines("%s\n" % l for l in lines)
    print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))


def evaluate_lexicon():
    ovo = OvO('data/polyglot.pkl', 'data/train.txt')
    train_trees = ListTrees('data/train.txt')

    lex = PLexicon()
    lex.learn(train_trees.rules_unchomsky)

    par = PCFGParser()
    par.count_rules(train_trees.rules)

    print('Nb rules: {}'.format(len(par.non_terminal_rules_counts.keys())))
    print('Nb terminal rules: {}'.format(len(par.lexicon_rules_counts.keys())))
    y_true = []
    y_pred = []
    test_trees = ListTrees('data/test.txt')

    for (c, tree) in enumerate(test_trees):
        print('{0:0.2f}% done'.format(100*c/len(test_trees)))
        tree.un_chomsky_normal_form(unaryChar='&')
        test_sentence = ovo.modify_tokens(list(tree.flatten()), False)
        parsed_labels = lex.predict_sentence(test_sentence)
        print(parsed_labels)
        print(leaves_labels(tree))
        y_true.extend(leaves_labels(tree))
        y_pred.extend(parsed_labels)

        print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))

    print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))
    
#evaluate_parser('data/evaluation_data.parser_output')
