from mnist_loader import Loader
from network import Neural_Network

loader = Loader()
train_data, val_data, test_data = loader.main()

sizes = [784,30,10]
network = Neural_Network(sizes)
network.SGD(train_data,30,10,3,test_data)
