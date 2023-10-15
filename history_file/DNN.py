"""
update log:添加批处理，超参数batch_size
"""
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist


class FullyConnectedLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.biases = np.zeros(output_size)
        self.input = None

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(output_gradient, self.weights.T)
        weights_gradient = np.dot(self.input.T, output_gradient)
        biases_gradient = np.sum(output_gradient, axis=0)

        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input > 0)


class SoftmaxWithCrossEntropy:
    def forward(self, input):
        exp_input = np.exp(input)
        self.softmax = exp_input / np.sum(exp_input, axis=1, keepdims=True)
        return self.softmax
    def backward(self, output_gradient, y):
        num_samples = y.shape[0]
        output_gradient[np.arange(num_samples), y.argmax(axis=1)] -= 1
        return output_gradient / num_samples


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc_layer1 = FullyConnectedLayer(input_size, hidden_size)
        self.relu = ReLU()
        self.fc_layer2 = FullyConnectedLayer(hidden_size, output_size)
        self.softmax_with_ce = SoftmaxWithCrossEntropy()

    def forward(self, input):
        hidden_output = self.fc_layer1.forward(input)
        relu_output = self.relu.forward(hidden_output)
        output = self.fc_layer2.forward(relu_output)
        return output

    def backward(self, output_gradient, learning_rate):
        relu_gradient = self.fc_layer2.backward(output_gradient, learning_rate)
        hidden_gradient = self.relu.backward(relu_gradient, learning_rate)
        self.fc_layer1.backward(hidden_gradient, learning_rate)

    def train(self, X, y, num_epochs, learning_rate, batch_size):
        loss_history = []

        for epoch in range(num_epochs):
            batch_mask = np.random.choice(X.shape[0], batch_size)
            x_batch = X[batch_mask]
            y_batch = y[batch_mask]
            output = self.forward(x_batch)
            loss = self.calculate_loss(output, y_batch)
            loss_history.append(loss)

            output_gradient = self.softmax_with_ce.backward(output, y_batch)
            self.backward(output_gradient, learning_rate)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss:.4f}')

        return loss_history

    def calculate_loss(self, output, y):
        num_samples = y.shape[0]
        class_probs = output[np.arange(num_samples), np.argmax(y, axis=1)]
        # 将class_probs限制在一个小的非零范围内
        class_probs = np.clip(class_probs, 1e-7, 1 - 1e-7)
        log_probs = -np.log(class_probs)
        loss = np.sum(log_probs) / num_samples
        return loss


(x_train, y_train), (x_test, y_test) = \
    load_mnist(normalize=True, one_hot_label=True)


# 定义超参数
input_size = 784
hidden_size = 100
output_size = 10
num_epochs = 1200
learning_rate = 0.01
batch_size = 500
# 创建神经网络对象
model = NeuralNetwork(input_size, hidden_size, output_size)
# 在训练集上训练神经网络
loss_history = model.train(x_train, y_train, num_epochs, learning_rate, batch_size)
# 绘制训练过程中loss的变化
plt.plot(range(num_epochs), loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()
# 在测试集上计算准确率
y_pred = np.argmax(model.forward(x_test), axis=1)
accuracy = np.mean(y_pred == np.argmax(y_test, axis=1))
print(f'Test Accuracy: {accuracy:.4f}')
