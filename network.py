import numpy as np
from loader import Loader

class Neural_Net:
    def __init__(self, sizes, eta, mini_batch_size, epochs):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.train_data, self.val_data ,self.test_data = Loader().getData()
        self.SGD(eta, mini_batch_size, epochs)

    def SGD(self, eta, mini_batch_size, epochs):
        n = len(self.train_data)
        n_test = len(self.test_data)
        for j in range(epochs):
            np.random.shuffle(self.train_data)
            mini_batches = [self.train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

            # TRAIN
            for mini_batch in mini_batches:
                self.updateMiniBatch(mini_batch, eta)

            # EVALUATE
            print ("{0}/{1} : {2}/{3}".format(j,epochs,self.evaluate(),n_test))

    def updateMiniBatch(self, mini_batch, eta):
        m = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/m)*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/m)*nw for w,nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        # FEED-FORWARD
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BACK-PROP
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def cost_derivative(self, output_activation, y):
        return (output_activation-y)

    def evaluate(self):
        test_result = [(np.argmax(self.feedForward(img)),label) for img,label in self.test_data]
        return sum(int(a == y) for a,y in test_result)

    def feedForward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


if __name__ == '__main__':
    sizes = [784, 30, 10]
    net = Neural_Net(sizes, 3.0, 10, 30)
