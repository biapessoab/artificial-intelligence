import numpy as np

class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1, n_epochs=1000):
        self.weights = np.random.rand(n_inputs)
        self.bias = np.random.rand()
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

    def train(self, inputs, labels):
        for epoch in range(self.n_epochs):
            for i in range(len(inputs)):
                prediction = self.predict(inputs[i])
                error = labels[i] - prediction
                self.weights += self.learning_rate * error * inputs[i]
                self.bias += self.learning_rate * error

    def predict(self, inputs):
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return 1 if weighted_sum > 0 else 0

# Função de teste para AND e OR
def test_perceptron(n_inputs, logic_gate):
    if logic_gate == "AND":
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([0, 0, 0, 1])
    elif logic_gate == "OR":
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        labels = np.array([0, 1, 1, 1])
    else:
        raise ValueError("Logic gate not recognized")

    perceptron = Perceptron(n_inputs)
    perceptron.train(inputs, labels)

    print(f"Testing {logic_gate} with {n_inputs} inputs:")
    for i in range(len(inputs)):
        result = perceptron.predict(inputs[i])
        print(f"Input: {inputs[i]} => Predicted: {result}")

test_perceptron(2, "AND")
test_perceptron(2, "OR")


# XOR
def test_xor(n_inputs):
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    labels = np.array([0, 1, 1, 0])

    perceptron = Perceptron(n_inputs)
    perceptron.train(inputs, labels)

    print(f"Testing XOR with {n_inputs} inputs:")
    for i in range(len(inputs)):
        result = perceptron.predict(inputs[i])
        print(f"Input: {inputs[i]} => Predicted: {result}")

test_xor(2)
