from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout
from PyQt5.QtCore import QThread, pyqtSignal
import pyqtgraph as pg
import sys

import numpy as np
from loader import Loader

class Graph(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('GRAPH')
        self.initUI()
        self.show()

    def initUI(self):
        self.layout = QHBoxLayout(self)
        self.plot1 = pg.PlotWidget(title='ACCURACY')
        self.plot2 = pg.PlotWidget(title='COST')
        self.layout.addWidget(self.plot1)
        self.layout.addWidget(self.plot2)

    def main(self):
        sizes = [784, 30, 10]
        eta = 3
        mini_batch_size = 10
        epochs = 30
        red_factor = 50         # REDUCTION IN SIZE OF DATA-SETS

        self.net = Neural_Net(sizes, eta, mini_batch_size, epochs, red_factor)
        self.net.trigger.connect(self.updateGraph)
        self.net.final_trigger.connect(self.finalizeGraph)

        # CURVE 1
        self.x1, self.y1 = [], []
        self.plot1.setXRange(0,epochs)
        self.plot1.setYRange(0,100)
        self.plot1.showGrid(x=True, y=True)
        self.curve1 = self.plot1.plotItem.plot()

        # CURVE 2
        self.y2 = []
        self.y2_max = 3
        self.plot2.setXRange(0,epochs)
        self.plot2.setYRange(0,self.y2_max)
        self.plot2.showGrid(x=True, y=True)
        self.curve2 = self.plot2.plotItem.plot()

        self.net.start()

    def updateGraph(self, idx, val, cost):
        self.x1.append(idx)
        self.y1.append(val)
        self.y2.append(cost)

        if self.y2_max < cost:
            self.y2_max = cost + 0.5
            self.plot2.setYRange(0, self.y2_max)

        self.curve1.setData(self.x1, self.y1)
        self.curve2.setData(self.x1, self.y2)

    def finalizeGraph(self):
        self.curve1.setPen('r')
        self.curve2.setPen('r')
        self.curve1.setData(self.x1, self.y1)
        self.curve2.setData(self.x1, self.y2)

class Neural_Net(QThread):
    trigger = pyqtSignal(int, float, float)
    final_trigger = pyqtSignal()
    def __init__(self, sizes, eta, mini_batch_size, epochs, red_factor):
        super().__init__()

        self.sizes = sizes
        self.eta = eta
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size

        self.biases = [np.random.randn(x,1) for x in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
        self.num_layers = len(sizes)

        self.train_data, self.val_data, self.test_data = self.loadData()
        self.train_data = self.train_data[:len(self.train_data)//red_factor]
        #self.test_data = self.test_data[:len(self.test_data)//red_factor]

    def run(self):
        self.SGD()

    def loadData(self):
        x = Loader()
        return x.getData()

    def SGD(self):
        n = len(self.train_data)
        n_test = len(self.test_data)
        for j in range(self.epochs):
            np.random.shuffle(self.train_data)
            mini_batches = [self.train_data[k:k+self.mini_batch_size]
                            for k in range(0, n, self.mini_batch_size)]
            # TRAIN
            for mini_batch in mini_batches:
                self.trainBatch(mini_batch)

            # EVALUATE
            accuracy, cost = self.evaluate()
            acc_percent = (accuracy/n_test)*100
            #print ("{0}/{1} : {2}/{3}".format(j, self.epochs, accuracy, n_test))
            #print ("{0:<3} -->  {1}".format(j+1, acc_percent))
            self.trigger.emit(j, acc_percent, cost)
        self.final_trigger.emit()

    def trainBatch(self, mini_batch):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backProp(x, y)
            nabla_b = [nb+dnb for nb,dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw,dnw in zip(nabla_w, delta_nabla_w)]
        self.biases = [b-(self.eta/self.mini_batch_size)*nb
                       for b,nb in zip(self.biases, nabla_b)]
        self.weights = [w-(self.eta/self.mini_batch_size)*nw
                        for w,nw in zip(self.weights, nabla_w)]

    def backProp(self, x, y):
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

        delta = self.costDerivative(activations[-1], y) * sigmoidPrime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoidPrime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def costDerivative(self, output_activations, y):
        return output_activations-y

    def evaluate(self):
        mse = lambda res,lab,n: (((res-lab)**2)/(2*n))  # MEAN-SQUARED ERROR
        test_results = [(np.argmax(self.feedForward(img)),label)
                        for img,label in self.test_data]
        n = len(test_results)
        return (sum(int(response == label) for response, label in test_results),
                sum(mse(response,label,n) for response, label in test_results))

    def feedForward(self, a):
        for b,w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

def sigmoid(z):
    return 1/(1+np.exp(-z))

def sigmoidPrime(z):
    return sigmoid(z)*(1-sigmoid(z))

if __name__ == '__main__':
    app = QApplication([])

    g = Graph()
    g.main()

    sys.exit(app.exec_())
