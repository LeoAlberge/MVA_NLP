import os


def read_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        return lines


def split_files(file_dir, main_file):
    lines = read_file(filepath=main_file)
    size = len(lines)
    train, dev, test = lines[0:int(size*8/10)], lines[int(size*8/10):int(size*9/10)], lines[int(size*9/10): int(size)]

    with open(os.path.join(file_dir, 'train.txt'), 'w') as f:
        f.writelines(train)

    with open(os.path.join(file_dir, 'dev.txt'), 'w') as f:
        f.writelines(dev)

    with open(os.path.join(file_dir, 'test.txt'), 'w') as f:
        f.writelines(test)


split_files('data', 'data/sequoia-corpus.txt')
