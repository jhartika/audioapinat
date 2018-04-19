"""Train Gaussian Mixture Model and do predictions"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

def main():
    #flow
    #read data
    #split to classes
    target_dir = "gmm"

    train_data_file = "data/ext/train_data.npy"
    train_labels_file = "data/ext/train_labels.npy"
    test_data_file = "data/ext/test_data.npy"
    test_labels_file = "data/ext/test_labels.npy"

    estimators = dict((cov_type, GaussianMixture(n_components=30,
                   covariance_type=cov_type, max_iter=200, random_state=0))
                  for cov_type in ['spherical', 'diag', 'tied', 'full'])

    train_data = np.load(train_data_file)
    train_labels = np.load(train_labels_file)
    test_data = np.load(test_data_file)
    test_labels = np.load(test_labels_file)

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
    
    scores = np.zeros((len(test_data), n_classes))
    for i in range(0, n_classes) :
        scores[:, i] = gmms[i].score_samples(test_data)
        
    print(np.shape(scores))
    predictions = np.argmax(scores, axis=1)
    print(accuracy_score(test_labels, predictions))
    print(predictions)
    print(test_labels)


    # gmm = GaussianMixture(n_components=10, tol=1e-3, max_iter=100, n_init=1, verbose=1)
    
    # for gmm in estimators.values() :
    #     """currently fitting just one gmm for all training data"""
    #     """ Would it be useful to try and fit one for each class?"""
    #     gmm.fit(train_data, train_labels)
    #     predictions = gmm.predict(test_data)
    #     """n_estimators = len(estimators)"""
    #     print(gmm.get_params())
    #     print(test_labels)
    #     print(predictions)
    #     print(accuracy_score(test_labels, predictions))
    #     print(np.shape(test_labels))
    #     print(np.shape(predictions))
        



if __name__ == '__main__':
    main()
