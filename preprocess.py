"""Abstraction for feature extraction and data manipulation"""
import itertools
import json
import os
import random

import librosa
import numpy as np

SR_KEY = "SR"
LABELS_KEY = "LABELS"
FEATURE_KEY = "FEATURES"


def extract_features(data_dir: str, output_dir: str, test_file_dir: str) -> None:
    # Load and split to testing and training
    class_tree = _files_by_class(data_dir)
    metadata = {LABELS_KEY: _labels_to_numbers(class_tree)}
    train, test = _train_test_split(class_tree)

    train_data_spec, train_labels_spec = _form_data_array(data_dir, train, metadata, False)
    train_data_spec = mel_spectrogram(train_data_spec, 40, metadata)
    test_data_spec, test_labels_spec = _form_data_array(data_dir, test, metadata, False)
    test_data_spec = mel_spectrogram(test_data_spec, 40, metadata)
    
    test_file_data_spec = _process_files(data_dir, test, metadata)
    
    print("Saving results")
    _save_metadata("data/mel", metadata)
    _save_results("data/mel", "train", train_data_spec, train_labels_spec)
    _save_results("data/mel", "test", test_data_spec, test_labels_spec)
    save_file_results(test_file_dir, test_file_data_spec)

    # Process
    print("Processing training samples")
    train_data, train_labels = _form_data_array(data_dir, train, metadata, True)
    train_data = _mfcc_features(train_data, 20, metadata)
    print("Processing testing samples")
    test_data, test_labels = _form_data_array(data_dir, test, metadata, True)
    test_data = _mfcc_features(test_data, 20, metadata)
    # Some work which is redone, but cannot be bothered with quality of code
    test_file_data = _process_files(data_dir, test, metadata)
    

    # Save results
    print("Saving results")
    _save_metadata(output_dir, metadata)
    _save_results(output_dir, "train", train_data, train_labels)
    _save_results(output_dir, "test", test_data, test_labels)
    save_file_results(test_file_dir, test_file_data)

    
def mel_spectrogram(data: np.ndarray, count: int, metadata: {str: object}) -> np.ndarray:
    sr = metadata[SR_KEY]
    metadata[FEATURE_KEY] = f"Mfcc, {count}"
    rows = data.shape[0]
    res = np.zeros((rows, count, 44))

    for i in range(rows):
        s = librosa.feature.melspectrogram(data[i, :], sr, n_mels=40, )
        res[i, :, :] = s
    return res


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


def _fetch_data_for_class(root: str, files: [str], to_split: bool) -> (np.ndarray, int):

    fetched = []
    sr = 0
    for file_name in files:
        file_name = f'{root}/{file_name}'
        if to_split:
            contents, sr = _fetch_data(file_name)
            fetched.append(contents)
        else:
            contents, sr = _fetch_data(file_name, 1.0)
            fetched.append(contents)
    res = np.concatenate(fetched)
    return res, sr


def _labels_to_numbers(class_tree: {str: [str]}) -> {str: int}:
    return dict([(k, i) for i, k in enumerate(class_tree.keys())])


def _form_data_array(data_dir: str, class_tree: {str: [str]}, metadata, to_split: bool) -> (np.ndarray, np.ndarray, {str: object}):
    labels = []
    data = []
    mapping = metadata[LABELS_KEY]
    for label, files in class_tree.items():
        print("Reading files for", label)
        contents, sr = _fetch_data_for_class(data_dir, files, to_split)
        metadata[SR_KEY] = sr
        num_label = mapping[label]
        labels.append(np.repeat(num_label, contents.shape[0]))
        data.append(contents)

    labels = np.concatenate(labels)
    data = np.concatenate(data)
    return data, labels


def _mfcc_features(data: np.ndarray, count: int, metadata: {str: object}) -> np.ndarray:
    sr = metadata[SR_KEY]
    metadata[FEATURE_KEY] = f"Mfcc, {count}"
    rows = data.shape[0]
    res = np.zeros((rows, count))

    for i in range(rows):
        s = librosa.feature.melspectrogram(data[i, :], sr, hop_length=data.shape[1] + 1, n_fft=data.shape[1])
        res[i, :] = np.ravel(librosa.feature.mfcc(S=librosa.power_to_db(s), n_mfcc=count))
    return res


def _save_metadata(target_dir: str, metadata) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # Save the metadata
    file_name = f"{target_dir}/metadata.json"
    with open(file_name, 'w') as f:
        json.dump(metadata, f)


def _save_results(target_dir: str, prefix: str, data: np.ndarray, labels: np.ndarray):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    label_name = f"{target_dir}/{prefix}_labels"
    data_name = f"{target_dir}/{prefix}_data"

    np.save(data_name, data)
    np.save(label_name, labels)


def save_file_results(target_dir: str, data: {str: np.ndarray}) -> None:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for file, cont in data.items():
        np.save(f"{target_dir}/{file}", cont)


def _process_files(root: str, class_tree: {str: [str]}, metadata: {str: object}) -> {str: np.array}:
    res = {}
    files = itertools.chain.from_iterable(class_tree.values())
    for file in files:
        print(f"Processing {file}")
        data, sr = _fetch_data(f'{root}/{file}')
        metadata['SR'] = sr
        data = _mfcc_features(data, 20, metadata)
        res[file] = data
    return res


def main():
    data_dir = "data/raw"
    target_dir = "data/ext"
    test_files_dir = "data/test"
    random.seed(12345)
    extract_features(data_dir, target_dir, test_files_dir)


if __name__ == '__main__':
    main()
