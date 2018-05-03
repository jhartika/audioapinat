"""Train Gaussian Mixture Model and do predictions"""
import numpy as np
import os
import json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def main():
    target_dir = "gmm"
    metadata_filepath = "data/ext/metadata.json"
    train_file_dir = "data/train_files"
    train_file_names = os.listdir(train_file_dir)

    #load metadata json
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)
    n_classes = len(metadata['LABELS'])
    train_data = [[] for i in range(n_classes)]

    delta_train_files = []
    train_files = []
    for file_name in train_file_names :
        parts = file_name.split('_')
        if(len(parts) == 3) :
            delta_train_files.append(file_name)
        if(len(parts) == 2) :
            train_files.append(file_name)

    for (file_name, delta_file_name) in zip(train_files, delta_train_files) :
        parts = file_name.split('_')#Get speaker from filename
        # if(len(parts) != 3) : #data without deltas has 2 parts
        #     continue
        speaker = parts[0]
        data = np.load(f'{train_file_dir}/{file_name}')
        delta_data = np.load(f'{train_file_dir}/{delta_file_name}')
        #print(np.shape(data))
        length = min(len(data), len(delta_data))
        print(np.shape(data[:len(data)-1]))
        print(np.shape(delta_data))
        data=np.concatenate((data[:length], delta_data[:length]), axis=1)

        train_data[metadata['LABELS'][speaker]].append(data)

    gmm = GaussianMixture(n_components=2, tol=1e-5, max_iter=200, n_init=1, verbose=1)
    delta_gmms = []
    for i in range(n_classes) :
        train_data[i]=np.concatenate(train_data[i][:])
        print(np.shape(train_data[i]))
        gmm.fit(train_data[i])
        joblib.dump(gmm, f'{target_dir}/delta_gmm_{i}.pkl') 

    for i in range(0, n_classes) :
        delta_gmm = joblib.load(f'{target_dir}/delta_gmm_{i}.pkl') 
        delta_gmms.append(delta_gmm)

    # for i in range(0, n_classes) :
    #     for j in range(0, len(train_data[i])) :
    #         traain_data[i].append(train_data[i][j])
    


    # #old
    train_data_file = "data/ext/train_data.npy"
    train_labels_file = "data/ext/train_labels.npy"
    # test_data_file = "data/ext/test_data.npy"
    # test_labels_file = "data/ext/test_labels.npy"

    train_data = np.load(train_data_file)
    train_labels = np.load(train_labels_file)
    # test_data = np.load(test_data_file)
    # test_labels = np.load(test_labels_file)

    # """ Fitting of the GMMs """
    # #number of different speakers
    
    # gmm = GaussianMixture(n_components=1, tol=1e-3, max_iter=200, n_init=1, verbose=1)
    gmms = []
    for i in range(0, n_classes) :
        speaker_train_data = train_data[train_labels==i]
        print(np.shape(speaker_train_data))
        gmm.fit(speaker_train_data)
        joblib.dump(gmm, f'{target_dir}/gmm_{i}.pkl') 
    
    for i in range(0, n_classes) :
        gmm = joblib.load(f'{target_dir}/gmm_{i}.pkl') 
        gmms.append(gmm)

    """ Predict using the GMMs """
    metadata_filepath = "data/ext/metadata.json"
    test_file_dir = "data/test"
    test_file_names = os.listdir(test_file_dir)

    #load metadata json
    with open(metadata_filepath, 'r') as f:
        metadata = json.load(f)

    labels = []
    preds = []

    file_names = []
    delta_file_names = []
    
    for file_name in test_file_names :
        parts = file_name.split('_')
        if(len(parts) == 3) :
            delta_file_names.append(file_name)
        if(len(parts) == 2) :
            file_names.append(file_name)

    #Make prediction per file in test_file_dir
    #for file_name in test_file_names :
    for (file_name, delta_file_name) in zip(file_names, delta_file_names) :
        parts = file_name.split('_')#Get speaker from filename
        # if(len(parts) != 3) : #data without deltas has 2 parts
        #     continue
        
        data = np.load(f'{test_file_dir}/{file_name}')
        delta_data = np.load(f'{test_file_dir}/{delta_file_name}')

        length = min(len(data), len(delta_data))
        print(np.shape(data[:len(data)-1]))
        print(np.shape(delta_data))
        data=np.concatenate((data[:length], delta_data[:length]), axis=1)

        testscores = np.zeros((len(data), n_classes))
        #Score each sample in a file with all GMMs
        for i in range(0, n_classes) :
            testscores[:, i] = delta_gmms[i].score_samples(data)
        

        # testscores = np.zeros((len(data)+len(delta_data), n_classes))
        # #Score each sample in a file with all GMMs
        # for sample in data :
        #     for i in range(0, n_classes) :
        #         testscores[0:len(data), i] = gmms[i].score_samples(data)
        #         testscores[len(data):, i] = delta_gmms[i].score_samples(delta_data)
        #Predict label(highest scoring GMM index) for each sample
        predictions = np.argmax(testscores, axis=1)

        #Majority vote between predictions for the file
        prediction = np.bincount(predictions).argmax()

        #Gather predictions and correct labels for accuracy score
        preds.append(prediction)
        label = metadata['LABELS'][parts[0]]#Get label matching speaker
        labels.append(label)
        print(f'pred:{prediction}, label:{label}')
    #Print accuracy score
    print(accuracy_score(labels, preds))

if __name__ == '__main__':
    main()
