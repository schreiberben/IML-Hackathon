from features import *
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pandas as pd

FILE_NAME = 'ovo_model.sav'
"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Authors: Yonatan Chamudot, Jason Greenspan, 
         Moshe Tannenbaum, Ben Schreiber

===================================================
"""


class GitHubClassifier:

    def __init__(self):
        self.model = load(FILE_NAME)  # Load the already trained OVOClassifier object

    def classify(self, X):
        """
        Receives a list of m unclassified pieces of code, and predicts for each
        one the Github project it belongs to.
        :param X: a numpy array of shape (m,) containing the code segments (strings)
        :return: y_hat - a numpy array of shape (m,) where each entry is a number between 0 and 6
        0 - building_tool
        1 - espnet
        2 - horovod
        3 - jina
        4 - PuddleHub
        5 - PySolFC
        6 - pytorch_geometric
        """
        # print("\tMatrix...")
        mat = output_design_matrix(X)
        # print("\tPredict...")
        return self.model.predict(mat)


def train(X, y):
    """
    Trains and saves the multi-class classification model
    :param X: The training design matrix
    :param y: The training response vector
    """
    ovo_classifier = OneVsOneClassifier(SVC(C=50, random_state=5)).fit(X, y)
    dump(ovo_classifier, FILE_NAME)  # Save the ovo_classifier object to a file


def reduce_num_of_samples(design_mat, response_vec):
    """
    Reduce the design matrix to have 7000 samples per coding project, or all lines if the project has less than 7000
    :param design_mat: The (m X d) design matrix. Numpy Array
    :param response_vec: The numpy array representing the response vector
    :return: The new design matrix and response vector
    """
    output_design_mat = np.empty(shape=(0, 350)).ravel()
    output_response_vec = np.empty(shape=(0, 1)).ravel()
    ptr = 0
    for i in range(7):
        num_of_samples = np.count_nonzero(response_vec == i)
        size = min(7000, num_of_samples)
        indices = np.arange(start=ptr, stop=size)
        ptr += num_of_samples  # Advance the pointer to the next coding project
        output_design_mat = np.concatenate((output_design_mat, design_mat[indices]), axis=0)
        output_response_vec = np.concatenate((output_response_vec, response_vec[indices]), axis=0)
    return output_design_mat, output_response_vec


if __name__ == "__main__":
    print("Reading Data...")
    data = pd.read_csv('design_mat.csv', sep=',').dropna().to_numpy()
    print("Splitting...")
    train_x, test_x, train_y, test_y = train_test_split(data[:, 0], data[:, 1], test_size=0.25)
    train_x, train_y = reduce_num_of_samples(train_x, train_y)
    train_y = train_y.astype('int')
    print("Matrixing...")
    design_mat = output_design_matrix(train_x)
    print("Training...")
    train(design_mat, train_y)  # Train the model and save it to a file
    print("Classifying...")
    # Classify the test data
    classifier = GitHubClassifier()
    y_hat = classifier.classify(test_x[np.arange(5000)])
    print("Accuracy =", np.count_nonzero(y_hat == test_y[np.arange(5000)]))  # Output Results
