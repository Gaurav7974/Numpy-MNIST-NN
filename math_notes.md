# the math behind this notebook

quick notes on every formula used in `mnist_nn.ipynb` and why its there. not a textbook explanation, just enough to actually get it.

---

## 1. normalizing pixels (X / 255)

```
X = X / 255
```

pixel values are 0-255. dividing by 255 squishes them to 0-1. networks train way better on small numbers like this instead of big ones, gradients dont explode and weights update more smoothly. basically just makes the math behave.

---

## 2. forward prop — layer 1

```
Z1 = W1.X + b1
```

this is just a weighted sum. `W1` is (10, 784) so each of the 10 hidden neurons has 784 weights, one per pixel. multiply that by the image `X` (784, m) and add the bias `b1`, you get `Z1` which is (10, m) — a raw score for each of the 10 hidden neurons, for every image in the batch.

nothing fancy, its literally just `y = wx + b` at scale.

---

## 3. ReLU

```
A1 = max(0, Z1)
```

after computing Z1, we pass it through ReLU. all it does is kill negative numbers, turns them to 0, keeps positives as is.

why we need this: without an activation function like this, stacking layers is pointless, two linear layers back to back is still just one linear layer mathematically. ReLU adds the non-linearity that lets the network actually learn curvy/complex patterns instead of just straight lines.

why ReLU specifically and not something else: its cheap to compute and doesnt have the vanishing gradient problem that older activations like sigmoid have.

---

## 4. forward prop — layer 2

```
Z2 = W2.A1 + b2
```

same thing as layer 1 but now `A1` (the ReLU output) is the input, and `W2` is (10,10) since we're going from 10 hidden neurons to 10 output classes (digits 0-9).

---

## 5. softmax

```
A2 = exp(Z2) / sum(exp(Z2))
```

this turns the 10 raw output scores into probabilities that add up to 1. so instead of getting some random numbers like [2.1, -0.5, 5.7...] you get something like [0.05, 0.01, 0.80...] where you can just pick the highest one as the prediction.

the `- max(Z)` before the exp() is just to stop the numbers from overflowing when Z is large, doesnt change the actual math, just keeps python from breaking.

**important bit:** we do this normalization down `axis=0` and not `axis=-1`. our Z2 has shape (10, m) — 10 classes are rows, m images are columns. we want each column (each image) to sum to 1, not each row. get this axis wrong and the whole thing falls apart, model just ends up predicting the same digit for everything.

---

## 6. one-hot encoding

```
Y -> [0,0,0,1,0,0,0,0,0,0]  (this example is for digit 3)
```

the labels come in as plain numbers like 3, 7, 9. but the network outputs 10 probabilities, one per class. so we cant directly compare "3" to a list of 10 numbers. one-hot just turns the label into a 10-length vector thats all zeros except a 1 at the correct digit's position. now we can subtract it from the softmax output and get an actual error signal.

---

## 7. backprop — output layer

```
dZ2 = A2 - one_hot(Y)
```

this is the error at the output layer. its literally just "predicted probabilities minus what it should've been." if the network was 100% confident and correct, this would be all zeros, no error. simple and clean, this specific formula comes from combining softmax with cross-entropy loss, the derivatives cancel out nicely into just this subtraction.

```
dW2 = (1/m) * dZ2.A1.T
db2 = (1/m) * sum(dZ2, across images)
```

dW2 tells us how much to adjust each weight in layer 2 — its basically "how much did this weight contribute to the error." db2 is the same idea but for the bias, averaged across all images in the batch (the `1/m`).

**note:** db2 has to stay shape (10,1), one value per output neuron. if you sum without specifying the axis, you get a single number and every neuron gets the exact same bias update, which basically breaks the network's ability to tell classes apart.

---

## 8. backprop — hidden layer

```
dZ1 = W2.T.dZ2 * ReLU_derived(Z1)
```

this pushes the error backwards from layer 2 into layer 1. `W2.T.dZ2` spreads the output error back to each hidden neuron based on how much they contributed. then we multiply by `ReLU_derived(Z1)` because if a neuron was "off" (Z1 was negative, ReLU zeroed it), it had zero effect on the output, so it shouldnt get blamed for the error either.

```
ReLU_derived(Z) = Z > 0
```

just says: if Z was positive, the slope was 1 (pass the gradient through). if Z was negative, ReLU flattened it, slope is 0 (block the gradient). thats the derivative of ReLU basically.

```
dW1 = (1/m) * dZ1.X.T
db1 = (1/m) * sum(dZ1, across images)
```

same idea as dW2/db2 but for layer 1 now.

---

## 9. gradient descent update

```
W = W - alpha * dW
b = b - alpha * db
```

this is the actual "learning" step. we nudge every weight and bias a little bit in the opposite direction of the gradient (thats why its minus), because the gradient points toward increasing error, so going the opposite way reduces it.

`alpha` is the learning rate, controls how big that nudge is. too high and it overshoots/never converges, too low and it takes forever to learn anything. we used 0.1 here.

---

## 10. accuracy check

```
accuracy = correct predictions / total predictions
```

nothing mathy here, literally counting how many predictions matched the actual label and dividing by total. just a sanity check on how the model's doing.
