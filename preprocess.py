"""Abstraction for feature extraction and data manipulation"""
import os

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


if __name__ == '__main__':
    pass