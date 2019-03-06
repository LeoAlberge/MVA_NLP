from trees import ListTrees
from ovo import OvO
from PCGF import PCFGParser


def main():
    ovo = OvO('data/polyglot.pkl', 'data/train.txt')
    train_trees = ListTrees('data/train.txt')

    par = PCFGParser()
    par.count_rules(train_trees.rules)

    test_tree = ListTrees('data/test.txt')[0]
    test_tree.un_chomsky_normal_form()
    test_tree.pretty_print()

    test_sentence = ovo.modify_tokens(test_tree.flatten())
    parsed_test = par.cky(test_sentence)
    parsed_test.pretty_print()


if __name__ == '__main__':
        main()
