import matplotlib.pyplot as plt
import numpy as np

# Question 2a
def plot_noisy_sine(lower, upper, x_train, y_train, x_test, y_test):
    plt.plot(np.linspace(lower,upper), np.sin(np.linspace(lower,upper)))
    plt.scatter(x_train, y_train, label="train")
    plt.scatter(x_test, y_test, label="test")
    plt.legend()

def poly_design(x, k):
    x = np.asarray(x)
    return np.vstack([x**i for i in range(k+1)]).T

def poly(x, W):
    x = np.asarray(x)
    w = np.asarray(W).reshape(-1)
    k = w.size - 1
    return poly_design(x, k) @ w

def fit_poly(x_train, y_train, k):
    A = poly_design(x_train, k)
    b = y_train
    return np.linalg.solve(A.T @ A, A.T @ b).reshape(1, k+1)

def mse_poly(x, y, W):
    w = np.asarray(W).reshape(-1)
    k = w.size - 1
    return np.mean(np.square(poly(x, w) - y))

def plot_fitted_polynomial(lower, upper, x_train, y_train, x_test, y_test, W, test_mse, filename):
    w = np.asarray(W).reshape(-1)
    k = w.size - 1
    plot_noisy_sine(lower, upper, x_train, y_train, x_test, y_test)

    plt.plot(np.linspace(lower, upper), poly(np.linspace(lower, upper), w), label=f"Fitted polynomial; degree {k} (MSE={round(test_mse,2)})")
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_mse_vs_k(K, filename, runs=1):
    fig, axes = plt.subplots(1, runs, figsize=(5 * runs, 4), sharey=True)
    if runs == 1:
        axes = [axes]
    for run, ax in enumerate(axes, start=1):
        mse = []
        x = np.random.rand(25) * 4*np.pi
        y = np.sin(x) + np.random.normal(0,0.1,25)
        x_train, y_train = x[:15], y[:15]
        x_test, y_test = x[15:], y[15:]

        for k in range(1, K+1):
            W = fit_poly(x_train, y_train, k)
            mse.append(mse_poly(x_test, y_test, W))

        ax.plot(range(1, K+1), mse)
        ax.set_title(f"run {run}")
        ax.set_xlabel("k (polynomial degree)")
        ax.set_yscale("log")
    axes[0].set_ylabel("MSE (log scale)")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

def ridge_fit_poly(x_train, y_train, k, lamb):
    X = poly_design(x_train, k)                 # (N, k+1)
    Y = y_train.reshape(-1, 1)                  # (N, 1)
    A = X.T @ X + lamb * np.eye(X.shape[1])     # (k+1, k+1)
    W = (Y.T @ X) @ np.linalg.inv(A)            # (1, k+1)
    return W

def grid_search(x_train, y_train, x_test, y_test, filename=None, ax=None, title=None):
    mse = np.zeros((20,20))
    lambdas = 10**np.linspace(-5, 0, 20)
    ks = np.arange(1, 21)
    for k in range(1,21):
        for i, lamb in enumerate(lambdas):
            W = ridge_fit_poly(x_train, y_train, k, lamb)
            mse[k-1, i] = np.log10(mse_poly(x_test,y_test,W))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    im = ax.imshow(mse, aspect="auto", origin="lower")
    ax.set_xlabel("lambda")
    ax.set_ylabel("k")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(lambdas)))
    ax.set_xticklabels([f"{l:.1e}" for l in lambdas], rotation=45)
    ax.set_yticks(np.arange(len(ks)))
    ax.set_yticklabels(ks)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if filename:
        fig.savefig(filename)
        plt.close(fig)
    best_idx = np.unravel_index(np.argmin(mse), mse.shape)
    best_k = best_idx[0] + 1
    best_lamb = lambdas[best_idx[1]]
    return best_k, best_lamb

def perform_cv(x, y, k, lamb, folds):
    N = len(x)
    assert N == len(y), "x and y need to be the same size"
    assert N % folds == 0, "folds need to be a divisor of the size of x"

    ans = []

    for fold in range(folds):
        start = fold * (N // folds)
        end = start + N // folds

        x_train = np.concatenate([x[:start], x[end:]])
        y_train = np.concatenate([y[:start], y[end:]])

        x_test = x[start:end]
        y_test = y[start:end]

        W = ridge_fit_poly(x_train, y_train, k, lamb)
        ans.append(mse_poly(x_test,y_test, W))
    return sum(ans) / folds

def plot_cv(k, lamb, filename):
    divisors = [i for i in range(2, 121) if 120 % i == 0]
    di = {divisor : [] for divisor in divisors}

    for i in range(100):
        x = np.random.rand(120) * 4*np.pi
        y = np.sin(x) + np.random.normal(0,0.1,120)
        for folds in divisors:
            ans = perform_cv(x, y, k, lamb, folds)
            di[folds].append(ans)

    avg = []
    stds = []
    for fold in divisors:
        avg.append(np.mean(di[fold]))
        stds.append(np.std(di[fold]))

    avg = np.array(avg)
    stds = np.array(stds)

    lower = np.maximum(avg - stds, 0)
    plt.plot(divisors, avg)
    plt.plot(divisors, avg + stds, 'g--')
    plt.plot(divisors, lower, 'g--')
    plt.xlabel("Number of folds")
    plt.ylabel("Cross-validated MSE")

    plt.savefig(filename)
    plt.close()


def main():
    # question 2a ; see also `plot_noisy_sine`
    x = np.random.rand(25) * 2*np.pi
    y = np.sin(x) + np.random.normal(0,0.1,25)
    x_train, y_train = x[:15], y[:15]
    x_test, y_test = x[15:], y[15:]
    lower = 0
    upper = 2*np.pi
    
    # question 2b
    k = 3
    W = fit_poly(x_train, y_train, k)
    test_mse = mse_poly(x_test, y_test, W) 
    plot_fitted_polynomial(lower, upper, x_train, y_train, x_test, y_test, W, test_mse, "question_2b.pdf")
    
    # question 2c
    plot_mse_vs_k(15, "question_2c_mse_vs_k.pdf", runs=3)

    k = 7 # see canvas for background on this choice
    lower = 0
    upper = 4*np.pi
    x = np.random.rand(25) * upper
    y = np.sin(x) + np.random.normal(0,0.1,25)
    x_train, y_train = x[:15], y[:15]
    x_test, y_test = x[15:], y[15:]

    W = fit_poly(x_train, y_train, k)
    test_mse = mse_poly(x_test, y_test, W)
    plot_fitted_polynomial(lower, upper, x_train, y_train, x_test, y_test, W, test_mse, "question_2c_polynomial.pdf")

    # question 2d ; see canvas for discussion
    fig, axes = plt.subplots(1, 3, figsize=(18, 4), sharey=True)
    best_k, best_lamb = grid_search(
        x_train, y_train, x_test, y_test, ax=axes[0], title="15 train / 10 test"
    )

    x = np.random.rand(1015) * upper
    y = np.sin(x) + np.random.normal(0,0.1,1015)
    x_train, y_train = x[:15], y[:15]
    x_test, y_test = x[15:], y[15:]

    grid_search(
        x_train, y_train, x_test, y_test, ax=axes[1], title="15 train / 1000 test"
    )

    x = np.random.rand(1510) * upper
    y = np.sin(x) + np.random.normal(0,0.1,1510)
    x_train, y_train = x[:1500], y[:1500]
    x_test, y_test = x[1500:], y[1500:]

    grid_search(
        x_train, y_train, x_test, y_test, ax=axes[2], title="1500 train / 10 test"
    )
    fig.tight_layout()
    fig.savefig("question_2d.pdf")
    plt.close(fig)

    # question 2e ; see canvas for discussion
    plot_cv(best_k, best_lamb, "question_2e.pdf")

if __name__ == "__main__":
    main()
