import numpy as np
import matplotlib.pyplot as plt

print("NumPy imported successfully")


# ============================================
# STEP 1: INPUT AND ACTUAL OUTPUT
# ============================================

X = np.array([[1, 0, 1, 0]], dtype=float)
Y = np.array([[1]], dtype=float)

print("Input:")
print(X)

print("Actual Output:")
print(Y)


# ============================================
# STEP 2: INITIALIZE WEIGHTS AND BIASES
# ============================================

np.random.seed(42)

# Input → Hidden Layer 1
W1 = np.random.randn(4, 4) * 0.5
b1 = np.zeros((1, 4))

# Hidden Layer 1 → Hidden Layer 2
W2 = np.random.randn(4, 3) * 0.5
b2 = np.zeros((1, 3))

# Hidden Layer 2 → Output
W3 = np.random.randn(3, 1) * 0.5
b3 = np.zeros((1, 1))

print("\nWeights initialized successfully.")


# ============================================
# STEP 3: SIGMOID FUNCTION
# ============================================

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


# ============================================
# STEP 4: FORWARD PROPAGATION
# ============================================

def forward(X):

    # Hidden Layer 1
    Z1 = np.dot(X, W1) + b1
    A1 = sigmoid(Z1)

    # Hidden Layer 2
    Z2 = np.dot(A1, W2) + b2
    A2 = sigmoid(Z2)

    # Output Layer
    Z3 = np.dot(A2, W3) + b3
    A3 = sigmoid(Z3)

    return Z1, A1, Z2, A2, Z3, A3


# ============================================
# STEP 5: TRAINING PARAMETERS
# ============================================

learning_rate = 0.5
target_error = 0.0001
max_iterations = 10000


# ============================================
# STEP 6: TRAIN THE NETWORK
# ============================================

for iteration in range(max_iterations):

    # FORWARD PASS
    Z1, A1, Z2, A2, Z3, A3 = forward(X)

    # Calculate Loss
    loss = 0.5 * np.mean((Y - A3) ** 2)

    # BACKWARD PASS

    # Output Layer
    dA3 = A3 - Y
    dZ3 = dA3 * sigmoid_derivative(A3)
    dW3 = np.dot(A2.T, dZ3)
    db3 = dZ3

    # Hidden Layer 2
    dA2 = np.dot(dZ3, W3.T)
    dZ2 = dA2 * sigmoid_derivative(A2)
    dW2 = np.dot(A1.T, dZ2)
    db2 = dZ2

    # Hidden Layer 1
    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * sigmoid_derivative(A1)
    dW1 = np.dot(X.T, dZ1)
    db1 = dZ1

    # UPDATE WEIGHTS AND BIASES
    W3 = W3 - learning_rate * dW3
    b3 = b3 - learning_rate * db3

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    # PRINT TRAINING PROGRESS
    if iteration % 100 == 0:
        print(
            "Iteration:",
            iteration,
            "Loss:",
            loss
        )

    # STOP TRAINING
    if loss < target_error:
        print(
            "\nTraining stopped at iteration:",
            iteration
        )
        print(
            "Final Loss:",
            loss
        )
        break


# ============================================
# STEP 7: FINAL PREDICTION
# ============================================

_, A1, _, A2, _, output = forward(X)

print("\n================================")
print("FINAL RESULTS")
print("================================")

print("\nInput:")
print(X)

print("\nActual Output:")
print(Y)

print("\nPredicted Output:")
print(output)

print("\nRounded Output:")
print(np.round(output))


# ============================================
# STEP 8: DISPLAY HIDDEN LAYER OUTPUTS
# ============================================

print("\n================================")
print("HIDDEN LAYER OUTPUTS")
print("================================")

print("\nHidden Layer 1 Output:")
print(A1)

print("\nHidden Layer 2 Output:")
print(A2)

print("\nFinal Output:")
print(output)


# ============================================
# STEP 9: VISUALIZE PREDICTION
# ===========================================3D

plt.figure(figsize=(8, 5))

plt.bar(
    ["Actual", "Predicted"],
    [Y[0][0], output[0][0]]
)

plt.ylabel("Output")
plt.title("Neural Network Prediction")

plt.show()
