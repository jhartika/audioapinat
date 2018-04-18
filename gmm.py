"""Train Gaussian Mixture Model and do predictions"""
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score

def main():
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
    # gmm = GaussianMixture(n_components=10, tol=1e-3, max_iter=100, n_init=1, verbose=1)
    for gmm in estimators.values() :
        """currently fitting just one gmm for all training data"""
        """ Would it be useful to try and fit one for each class?"""
        gmm.fit(train_data, train_labels)
        predictions = gmm.predict(test_data)
        """n_estimators = len(estimators)"""
        print(gmm.get_params())
        print(test_labels)
        print(predictions)
        print(accuracy_score(test_labels, predictions))
        print(np.shape(test_labels))
        print(np.shape(predictions))



if __name__ == '__main__':
    main()
