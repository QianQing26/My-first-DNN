import numpy as np
from collections import OrderedDict
from GDoptimizer import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from Layers import *

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        # input: W*input+b
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y==t)/float(x.shape[0])

        return accuracy

    # def numerical_gradient(self, x, t):
    #     loss_W = lambda W: self.loss(x, t)
    #
    #     grad = {}
    #     grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
    #     grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
    #     grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
    #     grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
    #
    #     return grad

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grad = {}
        grad['W1'] = self.layers['Affine1'].dW
        grad['b1'] = self.layers['Affine1'].db
        grad['W2'] = self.layers['Affine2'].dW
        grad['b2'] = self.layers['Affine2'].db

        return grad

    def train(self, x_train, t_train, x_test, t_test, \
              num_iter=5000, batch_size=16, lr=0.01, optimizer=Adam()):

        train_loss_list = []
        train_acc_list = []
        test_acc_list = []

        train_size = x_train.shape[0]
        iter_per_epoch = max(train_size/batch_size, 1)

        for i in tqdm(range(num_iter)):
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            grad = self.gradient(x_batch, t_batch)
            params = self.params
            optimizer.update(params, grad)

            loss = self.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            if i % iter_per_epoch == 0:
                train_acc = self.accuracy(x_train, t_train)
                test_acc = self.accuracy(x_test, t_test)
                train_acc_list.append(train_acc)
                test_acc_list.append(test_acc)
                print("\nEpochs:%d/%d    Loss:%.6f" % (i // iter_per_epoch, num_iter // iter_per_epoch, loss))
                print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

        plt.plot(range(num_iter), train_loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

        markers = {'train': 'o', 'test': 's'}
        x = np.arange(len(train_acc_list))
        plt.plot(x, train_acc_list, label='train acc')
        plt.plot(x, test_acc_list, label='test acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()
