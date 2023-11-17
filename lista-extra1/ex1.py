import numpy as np

# Função sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Função ReLU
def relu(x):
    return np.maximum(0, x)

# Derivada ReLU
def relu_derivative(x):
    return np.where(x > 0, 1, 0)



# Inicialização dos pesos da rede neural
def initialize_weights(input_size, hidden_size, output_size):
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
    return weights_input_hidden, weights_hidden_output

def train_neural_network(inputs, outputs, learning_rate, epochs):
    input_size = len(inputs[0])
    hidden_size = 4 
    output_size = len(outputs[0])

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        total_error = 0
        for i in range(len(inputs)):
            input_layer = inputs[i]
            hidden_layer_input = np.dot(input_layer, weights_input_hidden)
            hidden_layer_output = relu(hidden_layer_input)

            output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
            predicted_output = sigmoid(output_layer_input)

            error = outputs[i] - predicted_output
            total_error += np.mean(np.abs(error))

            output_error = error * sigmoid_derivative(predicted_output)
            hidden_error = output_error.dot(weights_hidden_output.T) * relu_derivative(hidden_layer_output)

            weights_hidden_output += hidden_layer_output.reshape(-1, 1) * output_error * learning_rate
            weights_input_hidden += input_layer.reshape(-1, 1) * hidden_error * learning_rate

        if epoch % 1000 == 0:
            print(f"Epoch {epoch}, Error: {total_error}")

    return weights_input_hidden, weights_hidden_output



# Função para prever com Sigmoid
def predict_sigmoid(input_data, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output

# Função para prever com ReLU
def predict_relu(input_data, weights_input_hidden, weights_hidden_output):
    hidden_layer_input = np.dot(input_data, weights_input_hidden)
    hidden_layer_output = relu(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    return predicted_output



# Escolher número de entradas
def get_user_input():
    num_inputs = int(input("Digite o número de entradas desejado: "))
    if num_inputs < 1:
        print("Número de entradas inválido. Deve ser pelo menos 1.")
        return get_user_input()
    return num_inputs

num_inputs = get_user_input()



# Definir os conjuntos de treinamento para AND OR e XOR
and_inputs = np.random.randint(2, size=(4, num_inputs))
and_outputs = np.array([np.all(i) for i in and_inputs]).reshape(-1, 1)

or_inputs = np.random.randint(2, size=(4, num_inputs))
or_outputs = np.array([np.any(i) for i in or_inputs]).reshape(-1, 1)

xor_inputs = np.random.randint(2, size=(4, num_inputs))
xor_outputs = np.logical_xor(xor_inputs[:, 0], xor_inputs[:, 1]).reshape(-1, 1)

# Parâmetros de treinamento
learning_rate = 0.1
epochs = 10000



# Treinamento
and_weights_input_hidden, and_weights_hidden_output = train_neural_network(and_inputs, and_outputs, learning_rate, epochs)

or_weights_input_hidden, or_weights_hidden_output = train_neural_network(or_inputs, or_outputs, learning_rate, epochs)

xor_weights_input_hidden, xor_weights_hidden_output = train_neural_network(xor_inputs, xor_outputs, learning_rate, epochs)


# Resultados Sigmoid
print("\nAND (Sigmoid):")
for i in range(len(and_inputs)):
    prediction = predict_sigmoid(and_inputs[i], and_weights_input_hidden, and_weights_hidden_output)
    print(f"AND({and_inputs[i]}) = {prediction}")

print("\nOR (Sigmoid):")
for i in range(len(or_inputs)):
    prediction = predict_sigmoid(or_inputs[i], or_weights_input_hidden, or_weights_hidden_output)
    print(f"OR({or_inputs[i]}) = {prediction}")

print("\nXOR (Sigmoid):")
for i in range(len(xor_inputs)):
    prediction = predict_sigmoid(xor_inputs[i], xor_weights_input_hidden, xor_weights_hidden_output)
    print(f"XOR({xor_inputs[i]}) = {prediction}")


# Resultados ReLu
print("\nAND (ReLU):")
for i in range(len(and_inputs)):
    prediction = predict_relu(and_inputs[i], and_weights_input_hidden, and_weights_hidden_output)
    print(f"AND({and_inputs[i]}) = {prediction}")

print("\nOR (ReLU):")
for i in range(len(or_inputs)):
    prediction = predict_relu(or_inputs[i], or_weights_input_hidden, or_weights_hidden_output)
    print(f"OR({or_inputs[i]}) = {prediction}")

print("\nXOR (ReLU):")
for i in range(len(xor_inputs)):
    prediction = predict_relu(xor_inputs[i], xor_weights_input_hidden, xor_weights_hidden_output)
    print(f"XOR({xor_inputs[i]}) = {prediction}")