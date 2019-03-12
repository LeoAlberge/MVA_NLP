from trees import ListTrees, leaves_labels, remove_leaves
from ovo import OvO
from PCGF import PCFGParser, PLexicon
from evaluation import accuracy

def main():
    ovo = OvO('data/polyglot.pkl', 'data/train.txt')
    train_trees = ListTrees('data/train.txt')


    par = PCFGParser()
    par.count_rules(train_trees.rules)

    print('Nb rules: {}'.format(len(par.non_terminal_rules_counts.keys())))
    print('Nb terminal rules: {}'.format(len(par.lexicon_rules_counts.keys())))
    y_true = []
    y_pred = []
    test_trees = ListTrees('data/test.txt')
    for (c, tree) in enumerate(test_trees):
        tree.un_chomsky_normal_form(unaryChar='&')
        print('True phrase: {}'.format(list(tree.flatten())))
        test_sentence = ovo.modify_tokens(list(tree.flatten()), True)
        print('OVO modified phrase: {}'.format(test_sentence))
        #test_sentence = list(tree.flatten())
        test_sentence = ["Le", "scandale", "a", "plusieurs", "clubs", "prend", "dans", "le", "championnat", "de", "Italie", "de", "football", "le", "championnat", "de", "Italie", "de", "football", '.']
        condition, parsed_tree = par.cky(test_sentence)
        break
        if condition:
            y_true.extend(leaves_labels(tree))
            y_pred.extend(leaves_labels(parsed_tree))
            print('Accuracy on dev set {}:'.format(accuracy(y_pred, y_true)))



if __name__ == '__main__':
        main()
