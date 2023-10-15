import numpy as np


class SGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate

    def update(self, params, grad):
        for key in params.keys():
            params[key] -= self.lr * grad[key]


class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:
    def __init__(self, lr=0.01, decay_rate=0.9):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Nesterov:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]


class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] * grads[key]
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)


class AdaMax:
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = {}
            self.v = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = np.maximum(self.beta2 * self.v[key], np.abs(grads[key]))
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            params[key] -= self.lr * m_hat / (self.v[key] + 1e-7)


class Adadelta:
    def __init__(self, rho=0.95, eps=1e-6):
        self.rho = rho
        self.eps = eps
        self.s = None
        self.dx = None

    def update(self, params, grads):
        if self.s is None:
            self.s = {}
            self.dx = {}
            for key, val in params.items():
                self.s[key] = np.zeros_like(val)
                self.dx[key] = np.zeros_like(val)
        for key in params.keys():
            self.s[key] = self.rho * self.s[key] + (1 - self.rho) * grads[key] ** 2
            delta_x = np.sqrt((self.dx[key] + self.eps) / (self.s[key] + self.eps)) * grads[key]
            self.dx[key] = self.rho * self.dx[key] + (1 - self.rho) * delta_x ** 2
            params[key] -= delta_x


class Adafactor:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.v = None
        self.m = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            self.m = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                self.m[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            v_hat = self.v[key] / (1 - self.beta2)
            m_hat = self.m[key] / (1 - self.beta1)
            params[key] -= (self.lr / (np.sqrt(v_hat) + self.eps)) * m_hat


class AMSGrad:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = None
        self.v = None
        self.v_hat = None
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = {}
            self.v = {}
            self.v_hat = {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                self.v_hat[key] = np.zeros_like(val)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            self.v_hat[key] = np.maximum(self.v_hat[key], self.v[key])
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v_hat[key] / (1 - self.beta2 ** self.t)
            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + 1e-7)


class RAdam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = None
        self.v = None
        self.rho_inf = None
        self.rho = None
        self.t = 0

    def update(self, params, grads):
        self.t += 1
        if self.m is None:
            self.m = {}
            self.v = {}
            self.rho_inf = 2 / (1 - self.beta2) - 1
            self.rho = self.rho_inf - 2 * self.t * self.beta2 ** self.t / (1 - self.beta2 ** self.t)
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * grads[key] ** 2
            if self.rho < 5:
                rho_hat = self.rho
            else:
                rho_hat = self.rho_inf
            if self.t > 1 and self.rho > 4:
                denominator = np.sqrt(self.v[key] / (1 - self.beta2 ** self.t)) + self.eps
                step_size = np.sqrt((1 - self.beta2 ** self.t) * (self.rho - 4) / (self.rho_inf - 4) * (
                            self.rho - 2) / self.rho * self.rho_inf)
                params[key] -= self.lr * step_size * self.m[key] / denominator
            else:
                params[key] -= self.lr * self.m[key] / (np.sqrt(self.v[key]) + self.eps)