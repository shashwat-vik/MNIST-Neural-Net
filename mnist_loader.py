import numpy as np
import gzip, pickle

class Loader:
    def __init__(self):
        self.train_data, self.val_data, self.test_data = None, None, None

    def unpack_data(self):
        with gzip.open('data/mnist.pkl.gz') as f:
            self.train_data, self.val_data, self.test_data = pickle.load(f, encoding='latin1')

    def reformat(self):
        self.train_data = list(zip(reshape_images_matrix(self.train_data[0]),vectorize_labels(self.train_data[1])))
        self.val_data = list(zip(reshape_images_matrix(self.val_data[0]),self.val_data[1]))
        self.test_data = list(zip(reshape_images_matrix(self.test_data[0]), self.test_data[1]))

    def main(self):
        self.unpack_data()
        self.reformat()
        return self.train_data,self.val_data, self.test_data

###################
# UTILITY FUNCTIONS

def vectorize_labels(labels):
    vec_labels = []
    for label in labels:
        x = np.zeros((10,1))
        x[label] = 1
        vec_labels.append(x)
    return vec_labels

def reshape_images_matrix(images):
    return [np.reshape(image,(784,1)) for image in images]

###################
