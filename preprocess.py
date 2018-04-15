"""Abstraction for feature extraction and data manipulation"""
import os
import random
random.seed(12345)

def extract_features(data_dir: str, output_dir: str) -> None:
    # First scan the file names and make labels
    pass


def _files_by_class(data_dir: str) -> {str: [str]}:
    class_tree = {}
    for file_name in os.listdir(data_dir):
        parts = file_name.split('_')
        if parts[0] not in class_tree:
            class_tree[parts[0]] = [file_name]
        else:
            class_tree[parts[0]].append(file_name)
    return class_tree


def _train_test_split(class_tree: {str: [str]}, split_ratio: float =0.8) -> ({str: [str]}, {str: [str]}):
    test = {}
    train = {}
    for label, files in class_tree.items():
        division_index = int(split_ratio * len(files)) + 1
        random.shuffle(files)
        train[label] = files[: division_index]
        test[label] = files[division_index:]
    return train, test


if __name__ == '__main__':
    class_tree = _files_by_class("data/raw")
    train, test = _train_test_split(class_tree)

    for k, v in class_tree.items():
        print(f"{k}: Total: {len(v)}, Train: {len(train[k])}, Test: {len(test[k])}, Ratio {len(test[k])/len(train[k])}")
