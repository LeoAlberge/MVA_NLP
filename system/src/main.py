from trees import ListTrees, leaves_labels, remove_leaves
from ovo import OvO
from PCGF import PCFGParser, PLexicon
import argparse


parser = argparse.ArgumentParser("PCFG")
parser.add_argument("--sentence", help="A sentence to parse", type=str)
parser.add_argument("--ovo", help="Out of vocabulary module, True for mixed strategy, False for evenshtein only", type=bool, default=True)
parser.add_argument("--remove_rules", help="Remove rules that occur only once", type=bool, default=True)
parser.add_argument("--verbose", help="Show complementary information", type=bool, default=False)
args = parser.parse_args()


def main():

    if args.verbose:
        print('Loading OVO..')

    ovo = OvO('src/data/polyglot.pkl', 'src/data/train.txt')

    if args.verbose:
        print('Loading done')

    if args.verbose:
        print('Training parser..')

    train_trees = ListTrees('src/data/train.txt')
    par = PCFGParser()
    par.count_rules(train_trees.rules)
    lex = PLexicon()
    lex.learn(train_trees.rules_unchomsky)

    if args.verbose:
        print('Training done')

    sentence = args.sentence.split(' ')
    if args.verbose:
        print(sentence)
    test_sentence = ovo.modify_tokens(sentence, levenshtein_only=not args.ovo, verbose=args.verbose)

    if args.verbose:
        print(test_sentence)

    condition, parsed_tree = par.cky(test_sentence)

    if not condition:
        print('The sentence is not parsable')
    else:
        parsed_tree.pretty_print()


if __name__ == '__main__':
        main()
