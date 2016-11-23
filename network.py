import numpy as np

class Neural_Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.num_layers = len(sizes)

    def conscious_reply(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w,a)+b)
        return a

    def SGD(self, train_data, epochs, mini_batch_size, etl, test_data):
        if test_data: n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            # TRAIN
            np.random.shuffle(train_data)
            mini_batches = [train_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, etl)

            # EVALUATE
            if test_data:
                print ("Epoch {0}/{1} : {2}/{3}".format(j,epochs,self.evaluate(test_data),n_test))

    def update_mini_batch(self, mini_batch, etl):
        m = len(mini_batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backpropogate(x,y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(etl/m)*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(etl/m)*nw for w,nw in zip(self.weights, nabla_w)]

    def backpropogate(self, x, y):
        # RIPPLE-FORWARD
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BACK_PASS
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

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.conscious_reply(pic)),label) for pic,label in test_data]
        return sum(int(reply==label) for reply,label in test_results)

###################
# UTILITY FUNCTIONS

def sigmoid(z):
    return 1/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z))

###################
