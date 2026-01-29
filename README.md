# neural-network-from-scratch
Feedforward neural network implemented from scratch using NumPy, with manual forward propagation, backpropagation, and analysis of training dynamics on a synthetic dataset.

Neural Network From Scratch (NumPy)

This repository contains a single-file implementation of a feedforward neural network built from scratch using NumPy.
The project focuses on explicitly implementing forward propagation, backpropagation, and gradient descent without relying on high-level machine learning frameworks.

The model is trained on a synthetic 2D classification dataset to analyze learning behavior, loss convergence, and decision boundary formation.

#What This Project Demonstrates

* Manual implementation of a feedforward neural network

* Forward propagation through a hidden layer

* Backpropagation using the chain rule

* Gradient descent–based parameter updates

* Training loss convergence

* Learned non-linear decision boundary

This project prioritizes clarity and understanding over performance or scalability.

# Model Overview

* Architecture: 2 → 4 → 2 (fully connected)

* Hidden activation: Sigmoid

* Output activation: Softmax

* Loss function: Cross-entropy

* Optimization: Stochastic Gradient Descent (per-sample updates)

* Dataset: Synthetic 2D radial classification task


# REpository Structure
- **neural-network-from-scratch/**
      - [train.py]
      - [requirements.txt]

    - ## results/
      - decision_boundary.png
      - loss_curve.png
    - **README.md**

# How to Run

## Clone the repository:

git clone <your-repo-url>
cd neural-network-from-scratch


## Install dependencies:

pip install -r requirements.txt


## Run training:

python train.py


This will:

generate a synthetic dataset

train the neural network

save the loss curve and decision boundary plots in the results/ directory

# Results
## Training Loss

The loss decreases steadily over training epochs, indicating stable optimization and correct gradient computation.

## Decision Boundary

The learned decision boundary shows that the network successfully models a non-linear separation in the 2D input space. Empirical comparison across different epoch counts shows that learning converges early, with diminishing returns beyond a certain number of epochs.

# Notes

This implementation is intentionally not vectorized to keep gradient flow explicit and readable.

The code is written as an experiment script, not as a reusable ML library.

The goal is foundational understanding, not state-of-the-art performance.

# Possible Extensions

- Vectorized training loop

- ReLU activation instead of sigmoid

- Mini-batch gradient descent

- Deeper architectures

- Noisy or more complex datasets

Requirements

Python 3.9+

NumPy

Matplotlib

## Final note

This project is meant to demonstrate understanding, not novelty.
It serves as a transparent reference for how neural networks learn at a low level.
