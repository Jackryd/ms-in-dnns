import torch
import matplotlib.pyplot as plt

def poly(x, k):
    return torch.vstack([x**i for i in range(k)]).T

def poly_predict(x, W):
    A = poly(x, len(W))
    return A @ W

def poly_fit(x_train, y_train, k):
    A = poly(x_train,k)
    b = y_train
    return torch.linalg.solve(A.T @ A, A.T @ b)

def comparison_plot(x_train, y_train, params, title, filepath):
    x = torch.linspace(0, 2*torch.pi, 100)
    y_gd = poly_predict(x, params)
    y_exact = poly_predict(x, poly_fit(x_train, y_train, 4))

    plt.clf()
    plt.plot(x, torch.sin(x), label="true")
    plt.plot(x, torch.detach(y_gd).numpy(), label="GD")
    plt.plot(x, y_exact, label="exact")
    plt.scatter(x_train, y_train, label="train")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.savefig(filepath)
    plt.clf()

def plot_losses(losses, label, title, filepath):
    plt.plot(losses, label=label)
    plt.xlabel("training step")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.yscale("log")
    plt.savefig(filepath)
    plt.clf()

def train_model(x_train, y_train, params, train_steps, opt):
    losses = []
    for i in range(train_steps):
        opt.zero_grad()
        y_pred = poly_predict(x_train, params)
        loss = torch.mean((y_train - y_pred) ** 2)
        losses.append(loss.item())
        loss.backward()
        opt.step()
    return losses

def main():
    N_TRAIN = 15
    SIGMA_NOISE = 0.1

    torch.manual_seed(0xDEADBEEF)
    x_train = torch.rand(N_TRAIN) * 2 * torch.pi
    y_train = torch.sin(x_train) + torch.randn(N_TRAIN) * SIGMA_NOISE

    params = torch.ones(4, requires_grad=True)
    train_steps = 100
    opt = torch.optim.SGD([params], lr=5e-5) 
    sgd_losses = train_model(x_train, y_train, params, train_steps, opt)
    plot_losses(sgd_losses, "SGD", "SGD Loss vs Steps", "train1.pdf")
    print("Final loss (SGD):", sgd_losses[-1])
    comparison_plot(x_train, y_train, params, "SGD Fit vs Sine", "sin_comp1.pdf")

    # Computation of Hessian
    A = torch.vstack([x_train**i for i in range(4)]).T
    H = (2.0 / x_train.shape[0]) * (A.T @ A)
    eigvals = torch.linalg.eigvalsh(H)
    # Condition number of Hessian
    cond = (eigvals.max() / eigvals.min()).item()
    print("eigenvalues:", eigvals)
    print("condition number:", cond)

    params = torch.tensor([0.1**(k) for k in range(4)], requires_grad=True)
    opt = torch.optim.SGD([params], lr=1e-4, momentum=0.93)
    momentum_losses = train_model(x_train, y_train, params, train_steps, opt)
    plot_losses(momentum_losses, "SGD with Momentum", "Momentum Loss vs Steps", "train2.pdf")
    print("Final loss (Momenum):", momentum_losses[-1])
    comparison_plot(x_train, y_train, params, "Momentum Fit vs Sine", "sin_comp2.pdf")

    params = torch.tensor([0.1**(k) for k in range(4)], requires_grad=True)
    opt = torch.optim.Adam([params], lr=1e-2)
    adam_losses = train_model(x_train, y_train, params, train_steps, opt)
    plot_losses(adam_losses, "Adam", "Adam Loss vs Steps", "train3.pdf") 
    print("Final loss (Adam):", adam_losses[-1]) 
    comparison_plot(x_train, y_train, params, "Adam Fit vs Sine", "sin_comp3.pdf")

    params = torch.tensor([0.1**(k) for k in range(4)], requires_grad=True)
    opt = torch.optim.LBFGS([params], lr=2e-3)
    def closure():
        opt.zero_grad()
        y_pred = poly_predict(x_train, params)
        loss = torch.mean((y_pred-y_train)**2)
        loss.backward()
        return loss
    lbfgs_losses = [] 
    for i in range(train_steps):
        loss = opt.step(closure)
        lbfgs_losses.append(loss.item())
    plot_losses(lbfgs_losses, "LBFGS", "LBFGS Loss vs Steps", "train4.pdf")
    print("Final loss (LBFGS):", lbfgs_losses[-1]) 
    comparison_plot(x_train, y_train, params, "LBFGS Fit vs Sine", "sin_comp4.pdf")

if __name__ == "__main__":
    main()
