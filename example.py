import numpy as np

from tape import Tensor

np.random.seed(2147483647)
# XOR example using tensor class

data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])


data = Tensor(data, requires_grad=False)
targets = Tensor(targets, requires_grad=False)

# config
H = 5
epochs = 1000
learning_rate = 0.1

# params
W1 = np.random.randn(2, H) * np.sqrt(2 / 2) # Kaiming init
b1 = np.zeros(
    H,
)  # (4, 2) @ (2, 3) => (4, 3) + (3, )

W2 = np.random.randn(H, 1) * np.sqrt(2 / H) # Kaiming init
b2 = np.zeros(
    1,
)  # (4, 3) @ (3, 1) => (4, 1) + (1, )

W1, b1, W2, b2 = (
    Tensor(W1, requires_grad=True),
    Tensor(b1, requires_grad=True),
    Tensor(W2, requires_grad=True),
    Tensor(b2, requires_grad=True),
)


# forward pass
def forward(input, W1, W2, b1, b2):
    h = input @ W1 + b1
    act = h.relu()
    logits = act @ W2 + b2
    return logits


# loss function
def mse_loss(pred, true):
    loss = (true - pred) ** 2
    loss /= len(data.data)
    loss = loss.sum()
    return loss


# utils
def round(x):
    result = []
    for i in x:
        if i > 0.5:
            result.append(1)
        else:
            result.append(0)
    return result


if __name__ == "__main__":
    # train loop
    for epoch in range(epochs):
        # forward pass
        logits = forward(data, W1, W2, b1, b2)

        # loss
        loss = mse_loss(logits, targets)

        # zero grad
        W1.zero_grad()
        W2.zero_grad()
        b1.zero_grad()
        b2.zero_grad()

        # backprop
        loss.backward()

        # SGD
        W1.data += -learning_rate * W1.grad
        W2.data += -learning_rate * W2.grad
        b1.data += -learning_rate * b1.grad
        b2.data += -learning_rate * b2.grad

        # print loss and predictions
        if epoch % 100 == 0:
            print(f"epoch: {epoch}")
            print(f"Loss: {loss.data:.4f}")
            print()
            print("Predictions: ")
            print(round(forward(data, W1, W2, b1, b2).data))
            print()
            print(f"Actual: \n{targets.data}")
            print("+" * 20)
