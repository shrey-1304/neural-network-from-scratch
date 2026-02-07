"""
Single-file NumPy implementation of a feedforward neural network.

Purpose:
- Implement forward propagation and backpropagation manually
- Train a neural network on a synthetic 2D classification dataset
- Visualize learning via loss curve and decision boundary
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    # HYPERPARAMETERS
    m, h, n = 2, 4, 2     # input dim, hidden units, output classes
    lr = 0.01            
    epochs = 750        
    N = 300              

    np.random.seed(42)


    # SYNTHETIC DATA GENERATION
    X = np.random.randn(m, N)

    # Radial decision boundary (non-linear problem)
    r = np.sqrt(X[0]**2 + X[1]**2)
    labels = (r < 1.0).astype(int)

    # One-hot encoded targets
    Y = np.zeros((n, N))
    Y[labels, np.arange(N)] = 1

    # PARAMETER INITIALIZATION
    
    W1 = np.random.randn(h, m)
    b1 = np.random.randn(h, 1)
    W2 = np.random.randn(n, h)
    b2 = np.random.randn(n, 1)

    
    # ACTIVATION FUNCTIONS
    
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def softmax(z):
        z = z - np.max(z, axis=0, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

    
    # TRAINING LOOP
    loss_history = []

    for ep in range(epochs):
        total_loss = 0.0

        for i in range(N):
            x = X[:, i:i+1]
            y = Y[:, i:i+1]

            # ----- Forward Pass -----
            z1 = W1 @ x + b1
            a1 = sigmoid(z1)
            z2 = W2 @ a1 + b2
            y_hat = softmax(z2)

            # ----- Loss -----
            loss = -np.sum(y * np.log(y_hat + 1e-9))
            total_loss += loss

            # ----- Backward Pass -----
            dz2 = y_hat - y
            dW2 = dz2 @ a1.T
            db2 = dz2

            da1 = W2.T @ dz2
            dz1 = da1 * (a1 * (1 - a1))
            dW1 = dz1 @ x.T
            db1 = dz1

            # ----- Parameter Update -----
            W2 -= lr * dW2
            b2 -= lr * db2
            W1 -= lr * dW1
            b1 -= lr * db1

        loss_history.append(total_loss / N)

    
    # DECISION BOUNDARY
    xx, yy = np.meshgrid(
        np.linspace(-3, 3, 300),
        np.linspace(-3, 3, 300)
    )

    grid = np.vstack([xx.ravel(), yy.ravel()])
    Z = []

    for i in range(grid.shape[1]):
        xg = grid[:, i:i+1]
        a1 = sigmoid(W1 @ xg + b1)
        y_hat = softmax(W2 @ a1 + b2)
        Z.append(np.argmax(y_hat))

    Z = np.array(Z).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, levels=1, cmap="bwr", alpha=0.3)
    plt.scatter(X[0], X[1], c=labels, cmap="bwr", edgecolors="k", s=20)
    plt.axis("equal")
    plt.title("Neural Network Decision Boundary")
    plt.savefig("decision_boundary.png")
    plt.close()


    # LOSS CURVE
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.savefig("loss_curve.png")
    plt.close()

    print("Training complete. Results saved.")


if __name__ == "__main__":
    main()
