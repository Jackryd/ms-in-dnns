import numpy as np
import matplotlib.pyplot as plt

class NPLinear:
    def __init__(self, in_channels, output_channels):
        stdv = 1.0 / np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size=(output_channels, in_channels))
        self.b = np.random.uniform(-stdv, stdv, size=output_channels)

        self.W_grad = np.zeros_like(self.W)
        self.b_grad = np.zeros_like(self.b)

        self.x_cache = None

    def forward(self, x):
        self.x_cache = x
        return x @ self.W.T + self.b

    def backward(self, grad):
        x = self.x_cache
        self.b_grad = grad.sum(axis=0)
        self.W_grad = grad.T @ x
        return grad @ self.W 

    def gd_update(self, lr):
        self.W = self.W - self.W_grad * lr
        self.b = self.b - self.b_grad * lr

class NPMSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def forward(self, predictions, targets):
        self.pred = predictions
        self.target = targets
        return np.mean(np.square(predictions - targets))

    def backward(self):
        pred = self.pred
        target = self.target
        return (2 / pred.size) * (pred - target)

class NPReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, grads):
        return grads * self.mask

class NPModel:
    def __init__(self, layer_sizes):
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(NPLinear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(NPReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grads):
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def gd_update(self, lr):
        for layer in self.layers:
            if isinstance(layer, NPLinear):
                layer.gd_update(lr)

def main():
    model = NPModel([1, 32, 32, 1])
    loss = NPMSELoss()
    lr = 0.05
    exp_decay = 0.9995

    N_TRAIN = 100
    N_TEST = 1000
    SIGMA_NOISE = 0.1

    np.random.seed(0xDEADBEEF)
    x_train = np.random.uniform(low=-np.pi, high=np.pi, size=N_TRAIN)[:, None]
    y_train = np.sin(x_train) + np.random.randn(N_TRAIN, 1) * SIGMA_NOISE

    plt.scatter(x_train, y_train)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig("training_data.pdf")
    plt.clf()

    x = np.linspace(-np.pi, np.pi, 100).reshape(100, 1)
    y = np.sin(x)
    plt.plot(x, y, label="$\sin x$")
    plt.xlabel("x")
    plt.ylabel("y")

    def predict_and_plot(x,model,epoch):
        y_pred = model.forward(x)
        plt.plot(x, y_pred, label=f"Epoch: {epoch}", alpha=0.4)

    x_test = np.random.uniform(low=-np.pi, high=np.pi, size=N_TEST)[:, None]
    y_test = np.sin(x_test) + np.random.randn(N_TEST, 1) * SIGMA_NOISE
    epochs = 1000
    losses = []
    val_losses = []

    for epoch in range(epochs):
        pred = model.forward(x_train)
        l = loss.forward(pred, y_train)
        losses.append(l)
        grads = loss.backward()
        model.backward(grads)
        model.gd_update(lr)
        lr *= exp_decay
        val_losses.append(loss.forward(model.forward(x_test), y_test))
        if epoch % 200 == 0:
            predict_and_plot(x,model,epoch)
    plt.title("NN Fits Across Training")
    plt.legend()
    plt.savefig("nn_sin_comp.pdf")
    plt.clf()

    plt.plot(losses, label="train")
    plt.plot(val_losses, label="test")
    plt.title("Training vs Validation Loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.savefig("train_nn.pdf")
    plt.clf()
    print("final training loss:", losses[-1])
    print("final validation loss:", val_losses[-1])

if __name__ == "__main__":
    main()
