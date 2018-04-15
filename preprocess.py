"""Abstraction for feature extraction and data manipulation"""
import json
import os
import random

import librosa
import numpy as np

SR_KEY = "SR"
LABELS_KEY = "LABELS"
FEATURE_KEY = "FEATURES"

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


def _fetch_data(file: str, sample_length: float =0.1) -> (np.ndarray, int):
    contents, sr = librosa.load(file)

    # Calculate the new dimensions and truncate the tail
    samples = int(sr * sample_length)
    rows = contents.shape[0] // samples
    contents = contents[: samples * rows]
    contents = contents.reshape((rows, samples))

    return contents, sr


def _fetch_data_for_class(root: str, files: [str]) -> (np.ndarray, int):

    fetched = []
    sr = 0
    for file_name in files:
        file_name = f'{root}/{file_name}'
        contents, sr = _fetch_data(file_name)
        fetched.append(contents)
    res = np.concatenate(fetched)
    return res, sr


def _labels_to_numbers(class_tree: {str: [str]}) -> {str: int}:
    return dict([(k, i) for i, k in enumerate(class_tree.keys())])


def _form_data_array(data_dir: str, class_tree: {str: [str]}, metadata) -> (np.ndarray, np.ndarray, {str: object}):
    labels = []
    data = []
    mapping = metadata[LABELS_KEY]
    for label, files in class_tree.items():
        print("Reading files for", label)
        contents, sr = _fetch_data_for_class(data_dir, files)
        metadata[SR_KEY] = sr
        num_label = mapping[label]
        labels.append(np.repeat(num_label, contents.shape[0]))
        data.append(contents)

    labels = np.concatenate(labels)
    data = np.concatenate(data)
    return data, labels


def _mfcc_features(data: np.ndarray, count: int, metadata: {str: object}) -> np.ndarray:
    print("Calculating features")
    sr = metadata[SR_KEY]
    metadata[FEATURE_KEY] = f"Mfcc, {count}"
    rows = data.shape[0]
    res = np.zeros((rows, count))

    for i in range(rows):
        # res[i, :] = librosa.feature.mfcc(data[i, :], sr, spectr, n_mfcc=count)
        # Alternatively use above and just flatmap or something
        s = librosa.feature.melspectrogram(data[i, :], sr, hop_length=data.shape[1], n_fft=data.shape[1])
        res[i, :] = np.ravel(librosa.feature.mfcc(S=librosa.power_to_db(s), n_mfcc=count))
    return res


def _save_metadata(target_dir: str, metadata) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Save the metadata
    file_name = f"{target_dir}/metadata.json"
    with open(file_name, 'w') as f:
        json.dump(metadata, f)


def main():
    data_dir = "data/raw"
    target_dir = "data/ext"
    random.seed(12345)
    class_tree = _files_by_class(data_dir)
    metadata = {LABELS_KEY: _labels_to_numbers(class_tree)}
    train, test = _train_test_split(class_tree)
    test_data, test_labels = _form_data_array(data_dir, test, metadata)
    test_data = _mfcc_features(test_data, 20, metadata)
    _save_metadata(target_dir, metadata)


if __name__ == '__main__':
    main()
