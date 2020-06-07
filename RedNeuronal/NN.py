import numpy as numpy

# np.random.seed(10)


class NeuralNetwork:
    def __init__(self, n_inputs=1, hidden_layers=[], n_outputs=1):
        self.weights = []
        self.output_layers = []
        self.errors = []

        input_neurons = n_inputs
        for layer_neurons in hidden_layers:
            self.weights.append(numpy.random.rand(input_neurons, layer_neurons))
            input_neurons = layer_neurons
        self.weights.append(numpy.random.rand(input_neurons, n_outputs))

    def output_function(self, value, dx=False, method='sigmoid'):
        if method == 'sigmoid':
            if dx:
                return numpy.round(numpy.multiply(value, (1.0 - value)), 4)
            return numpy.round(1.0 / (1.0 + numpy.exp(-value)), 4)

    def forward(self, inputs):
        outputs = []
        for w in self.weights:
            activation = numpy.dot(inputs, w)
            output = self.output_function(activation)
            inputs = output.copy()
            outputs.append(output)
        self.output_layers = outputs.copy()
        return self.output_layers[-1]  # Return last element

    def backpropagation(self, X, y, output, eta=0.35):
        deltas = []
        error = 2 * (output - y)
        for layer, weight in zip(reversed(self.output_layers), reversed(self.weights)):
            dx = self.output_function(layer, True)
            delta = numpy.multiply(error, dx)
            error = numpy.dot(delta, weight.T)
            deltas.append(delta)

        deltas_weights = []
        for layer, delta in zip(reversed(self.output_layers[:-1]), deltas):
            correction = (-1 * eta) * numpy.dot(layer.T, delta)
            deltas_weights.append(correction)
        correction = (-1 * eta) * numpy.dot(X.T, deltas[-1])
        deltas_weights.append(correction)

        deltas_weights.reverse()
        for i in range(len(self.weights)):
            self.weights[i] = numpy.add(self.weights[i], deltas_weights[i])

    def trainStep(self, X, y):
        output = self.forward(X)
        self.backpropagation(X, y, output)

    def train(self, X, y, minError=1 / 1000, iteration=10000, eta=0.35):
        for i in range(iteration):
            print(f'It: {i}')
            output = self.forward(X)
            error = numpy.mean(numpy.square(output - y))
            print(f'Error: {error}')
            self.errors.append(error)
            if error <= minError:
                break
            self.backpropagation(X, y, output, eta)
        return error, i

