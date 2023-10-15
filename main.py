from network import TwoLayerNet
import GDoptimizer
from dataset.mnist import load_mnist


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)

    print(x_train.shape)

    num_iterations = 5000
    batch_size = 32
    learning_rate = 0.01
    optimizer = GDoptimizer.Adam()

    network = TwoLayerNet(784, 81, 10)
    network.train(x_train, t_train, x_test, t_test,
                  num_iter=num_iterations, batch_size=batch_size,
                  lr=learning_rate, optimizer=optimizer)