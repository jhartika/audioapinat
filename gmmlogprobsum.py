"""Train Gaussian Mixture Model and do predictions"""
import numpy as np
import os
import json
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def main():
    target_dir = "gmm"

    train_data_file = "data/ext/train_data.npy"
    train_labels_file = "data/ext/train_labels.npy"
    test_data_file = "data/ext/test_data.npy"
    test_labels_file = "data/ext/test_labels.npy"

    #Not used atm but could train several different GMMs
    estimators = dict((cov_type, GaussianMixture(n_components=30,
                   covariance_type=cov_type, max_iter=200, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

    train_data = np.load(train_data_file)
    train_labels = np.load(train_labels_file)
    test_data = np.load(test_data_file)
    test_labels = np.load(test_labels_file)

    """ Fitting of the GMMs """
    #number of different speakers
    n_classes = len(np.unique(test_labels))
    gmm = GaussianMixture(n_components=1, tol=1e-3, max_iter=200, n_init=1, verbose=1)
    gmms = []
    for i in range(0, n_classes) :
        speaker_train_data = train_data[train_labels==i]
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
    #Make prediction per file in test_file_dir
    for file_name in test_file_names :
        parts = file_name.split('_')#Get speaker from filename
        if(len(parts) != 2) : #data without deltas has 2 parts
            continue
        
        data = np.load(f'{test_file_dir}/{file_name}')
        testscores = np.zeros((len(data), n_classes))
        #Score each sample in a file with all GMMs
        for i in range(0, n_classes) :
            testscores[:, i] = gmms[i].score_samples(data)
        #Predict label(highest scoring GMM index) for each sample
        predictions = np.sum(testscores, axis=0)

        #Majority vote between predictions for the file
        prediction = predictions.argmax()

        #Gather predictions and correct labels for accuracy score
        preds.append(prediction)
        label = metadata['LABELS'][parts[0]]#Get label matching speaker
        labels.append(label)
        print(f'pred:{prediction}, label:{label}')
    #Print accuracy score
    print(accuracy_score(labels, preds))

if __name__ == '__main__':
    main()
