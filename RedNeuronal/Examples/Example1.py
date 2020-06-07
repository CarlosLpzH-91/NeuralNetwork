import numpy as np
from RedNeuronal import NN
from matplotlib import pyplot as plt

X = [[1, 0.25, -0.5]]  # 1 example, 3 neurons
y = [[1, -1, 0]]  # 1 example, 3 neurons

hidden_layers = [2]  # 1 hidden layer, 2 neurons

train_input = np.array(X)
train_output = np.array(y)

# Since sigmoid func can't produce negative or above 1 results.
min_value = train_output.min()
train_output = train_output - min_value
max_value = train_output.max()
train_output = train_output / max_value

neural_network = NN.NeuralNetwork(train_input.shape[1], hidden_layers, train_output.shape[1])

error, i = neural_network.train(train_input, train_output, minError=0.0011)
print(f'Final Error: {error}')
print(f'Iterations: {i}')

print(f'Final Output: \n{np.round(neural_network.output_layers[-1], 1)}')
for i, weight in enumerate(neural_network.weights):
    print(f'Final weight {i}:\n{np.round(weight, 4)}')

plt.plot(neural_network.errors)
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.show()