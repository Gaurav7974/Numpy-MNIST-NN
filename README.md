# numpy-mnist-nn

A 2-layer neural network for MNIST digit classification, built from scratch using **pure NumPy** — no PyTorch, no TensorFlow, no autograd. Forward pass, backward pass, and gradient descent are all implemented and derived by hand.

This was built to actually understand what's happening inside a neural network instead of calling `model.fit()`.

## Why this exists

Frameworks hide the math. This project doesn't. Every gradient in `backward_prop` is manually derived, and getting the shapes and broadcasting right (or wrong) has real, visible consequences on training — which is the whole point.

## Architecture

- **Input layer:** 784 units (28×28 flattened pixel images)
- **Hidden layer:** 10 units, ReLU activation
- **Output layer:** 10 units, Softmax activation (one per digit, 0-9)
- **Loss:** implicit cross-entropy via `A2 - one_hot(Y)` in the backward pass
- **Optimizer:** vanilla full-batch gradient descent (no momentum, no Adam)

```
X (784, m) -> W1, b1 -> ReLU -> W2, b2 -> Softmax -> predictions
```

## Results

Trained for 500 iterations at learning rate 0.1 on ~41,000 training images (1,000 held out for a dev/validation split).

| Split | Accuracy |
|-------|----------|
| Train | *fill in after your fixed run* |
| Dev   | *fill in after your fixed run* |

## Getting the data

This repo does **not** include the dataset. Download it from Kaggle's [Digit Recognizer competition](https://www.kaggle.com/competitions/digit-recognizer/data) and place `train.csv` in a `data/` folder at the repo root:

```
numpy-mnist-nn/
└── data/
    └── train.csv
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook mnist_nn.ipynb
```

Run all cells top to bottom. Training progress (iteration count + running accuracy) prints every 10 iterations.

## What I'd improve next

- Mini-batch gradient descent instead of full-batch (currently trains on all ~41k samples every step)
- A proper hidden layer size sweep — 10 hidden units is small and arbitrary
- L2 regularization / weight decay, since there's currently none
- Replace the dev-set accuracy check with a real held-out test set, not just a slice of train

## Notes

Built as part of a personal deep-dive into neural net fundamentals before moving to framework-based work. Bugs found and fixed along the way (wrong softmax normalization axis, collapsed bias gradients) are preserved in commit history rather than squashed — the debugging process is part of the point.
