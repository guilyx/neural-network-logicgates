import numpy as np 
import time

class NeuralNetwork():
    def __init__(self, learning_rate = None, synapses = None):

        if (synapses == None):
            self.synapses = 2 * np.random.random((2, 1)) - 1
        else: 
            self.synapses = synapses

        self.epoch = 0
        self.training_time = 0

        if learning_rate == None:
            self.learning_rate = 1
        else:
            self.learning_rate = learning_rate


    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    
    def tanh(self, z):
        return (2.0 / (1 + np.exp(-2*z))) - 1

    
    def tanh_prime(self, z):
        return 1.0 - self.tanh(z)**2


    def reshape(self, inputs):
        if (inputs[0].shape != self.synapses.shape):
            dim = inputs.shape
            self.synapses.resize(dim[0], dim[1])
        print("Synapses vector reshaped to the inputs' size..")

    
    def predict(self, inputs):
        inputs = inputs.astype(float)
        outputs = self.tanh(np.dot(inputs, self.synapses))
        return outputs

    
    def train(self, t_inputs, t_outputs, epochs):
        start_time = time.time()

        for iterations in range(epochs):

            self.epoch = iterations

            input_layer = t_inputs
            outputs = self.sigmoid(np.dot(input_layer, self.synapses))
            np.round(outputs)
            error = t_outputs - outputs

            adjustments = error * self.sigmoid_prime(outputs)
            self.synapses += np.dot(input_layer.T, adjustments) * self.learning_rate
        
        self.training_time = time.time() - start_time