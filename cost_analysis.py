from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import numpy as np
import time

class Graph(QWidget):
    def __init__(self, epochs):
        super().__init__()
        self.epochs = epochs
        self.initUI()
        self.show()

    def initUI(self):
        self.setWindowTitle('COST FUNCTION')
        self.layout = QVBoxLayout(self)

        self.plot = pg.PlotWidget()
        self.layout.addWidget(self.plot)
        self.curve = self.plot.plotItem.plot()

        self.x = np.arange(self.epochs+1)
        self.y = np.zeros(self.epochs+1)
        self.curve.setData(self.x, self.y)

        self.network()

    def network(self):
        self.neural_net = Neural_Net(self.epochs)
        self.neural_net.trigger.connect(self.updateGraph)
        self.neural_net.start()

    def updateGraph(self, idx, val):
        self.y[idx] = val
        self.curve.setData(self.x, self.y)

class Neural_Net(QThread):
    trigger = pyqtSignal(int, float)
    def __init__(self, xyz_epochs):
        super().__init__()
        self.xyz_epochs = xyz_epochs
        self.sizes = [1,1]
        self.num_layers = len(self.sizes)
        self.biases = [np.array([0.9])]
        self.weights = [np.array([[0.6]])]

    def run(self):
        train_data = [[np.array([[1]]),0] for i in range(100)]
        self.GD(train_data, 0.15, self.xyz_epochs)

    def sendUpdate(self, idx, val):
        self.trigger.emit(idx, val)

    def GD(self, train_data, eta, epochs):
        count = 0
        self.sendUpdate(count, self.output([[1]])[0][0])
        for j in range(epochs):
            count += 1
            self.train_batch(train_data, eta)
            self.sendUpdate(count, self.output([[1]])[0][0])
            time.sleep(0.01)

    def train_batch(self, batch, eta):
        m = len(batch)
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(eta/m)*nb for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(eta/m)*nw for w,nw in zip(self.weights, nabla_w)]

    def backprop(self, x, y):
        # FEED FORWARD
        activation = x
        activations = [x]
        zs = []
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # BACK-PROPAGATE
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

    def output(self, x):
        for b,w in zip(self.biases, self.weights):
            z = np.dot(w,x)+b
            x = sigmoid(z)
        return x

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    app = QApplication([])
    g = Graph(400)
    app.exec_()
