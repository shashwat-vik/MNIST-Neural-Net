import gzip, pickle
import numpy as np

class Loader:
    def __init__(self):
        self.train, self.val, self.test = None, None, None

    def unpackData(self):
        with gzip.open('data/mnist.pkl.gz') as f:
            self.train, self.val, self.test = pickle.load(f, encoding='latin1')

    def getData(self):
        self.unpackData()
        self.train = list(zip(reshape_images_vector(self.train[0]), vectorize_labels(self.train[1])))
        self.val = list(zip(reshape_images_vector(self.val[0]), self.val[1]))
        self.test = list(zip(reshape_images_vector(self.test[0]), self.test[1]))
        return self.train, self.val, self.test


def reshape_images_vector(images):
    reshaped = []
    for image in images:
        reshaped.append(np.reshape(image, (784,1)))
    return reshaped

def vectorize_labels(labels):
    vectored = []
    for label in labels:
        x = np.zeros((10, 1))
        x[label] = 1
        vectored.append(x)
    return vectored
