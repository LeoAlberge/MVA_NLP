from nltk.tree import Tree


def read_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return lines


def clean_labels(tree):
    """

    :param tree:
    :return:
    """
    if tree.label().find('-'):
        tree.set_label(tree.label().split('-')[0])
    for child in tree:
        if isinstance(child, Tree):
            clean_labels(child)


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


def remove_leaves(tree):
    """

    :param tree:
    :return:
    """
    if isinstance(tree, Tree) and not is_terminal(tree):
        for i in range(len(tree)):
            if isinstance(tree[i], Tree):
                tree[i] = remove_leaves(tree[i])
    elif isinstance(tree, Tree) and is_terminal(tree):
        tree.remove(tree[0])
        tree.append(tree.label())
        return tree
    return tree


def leaves_with_labels(tree, label=None):
    """
    Return list with leaves token and label(POS)
    :param tree
    :param label:
    :return:
    """
    leaves = []
    for child in tree:
        if isinstance(child, Tree):
            leaves.extend(leaves_with_labels(child, child.label()))
        else:
            leaves.append((child, label))
    return leaves


def leaves_labels(tree, label=None):
    """
    Return list with leaves token and label(POS)
    :param tree
    :param label:
    :return:
    """
    leaves = []
    for child in tree:
        if isinstance(child, Tree):
            leaves.extend(leaves_labels(child, child.label()))
        else:
            leaves.append(label)
    return leaves


class ListTrees(object):
    """
    Class to store trees and extract rules.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.trees = []
        self.lines = read_file(filepath)
        self.rules = []
        self.rules_unchomsky = []
        self.leaves_with_label = []
        for line in self.lines:
            tree = Tree.fromstring(line)[0]
            clean_labels(tree)
            self.leaves_with_label.extend(leaves_with_labels(tree))
            self.rules_unchomsky.extend(tree.productions())

            tree.collapse_unary(True, True, '&')
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

    def __len__(self):
        return len(self.trees)


