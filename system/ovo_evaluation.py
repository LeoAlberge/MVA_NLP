from ovo import OvO
from trees import ListTrees
from evaluation import accuracy

ovo = OvO('data/polyglot.pkl', 'data/train.txt')
train_trees = ListTrees('data/train.txt')
dev_trees = ListTrees('data/dev.txt')

vocab = train_trees.get_vocabulary()

train_ref = {}
for word, pos in train_trees.leaves_with_label:
    train_ref[word] = pos

print(train_ref)

y_true = []
y_pred = []

for word, pos in dev_trees.leaves_with_label:
    if word not in vocab:
        replaced = ovo.most_similar_levenshtein(word, 1)[0]
        pos_replaced = train_ref[replaced]
        true_pos = pos
        y_true.append(true_pos)
        y_pred.append(pos_replaced)

print('Accuracy POS tags: {}'.format(accuracy(y_pred, y_true)))