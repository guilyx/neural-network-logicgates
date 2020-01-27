import numpy as np 
import time

class Perceptron():
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
        return z * (1 - z)

    
    def tanh(self, z):
        return (2.0 / (1 + np.exp(-2*z))) - 1

    
    def tanh_prime(self, z):
        return 1.0 - z**2


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
            outputs = self.tanh(np.dot(input_layer, self.synapses))

            error = t_outputs - outputs

            adjustments = error * self.tanh_prime(outputs)
            self.synapses += np.dot(input_layer.T, adjustments) * self.learning_rate
        
        self.training_time = time.time() - start_time


class NeuralNetwork():
    def __init__(self, learning_rate = None):

        self.synapse_0 = 2 * np.random.random((2, 2)) - 1
        self.synapse_1 = 2 * np.random.random((2, 1)) - 1
        self.synapse_2 = 2 * np.random.random((1, 2)) - 1

        self.epoch = 0
        self.training_time = 0

        if learning_rate == None:
            self.learning_rate = 1
        else:
            self.learning_rate = learning_rate


    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))


    def sigmoid_prime(self, z):
        return z * (1 - z)

    
    def tanh(self, z):
        return (2.0 / (1 + np.exp(-2*z))) - 1

    
    def tanh_prime(self, z):
        return 1.0 - z**2


    def predict(self, inputs):
        inputs = inputs.astype(float)

        layer0 = inputs
        layer1 = self.tanh(np.dot(layer0, self.synapse_0))
        layer2 = self.tanh(np.dot(layer1, self.synapse_1))
        layer3 = self.tanh(np.dot(layer2, self.synapse_2))

        outputs = layer3
        return outputs

    
    def train(self, t_inputs, t_outputs, epochs, debug = False):
        start_time = time.time()
        
        bias_1 = bias_2 = bias_3 = 0

        for iterations in range(epochs):

            self.epoch = iterations

            # Forward Propagation
            input_layer = t_inputs
            layer0 = input_layer
            layer1 = self.tanh(np.dot(layer0, self.synapse_0) + bias_1)
            layer2 = self.tanh(np.dot(layer1, self.synapse_1) + bias_2)
            layer3 = self.tanh(np.dot(layer2, self.synapse_2) + bias_3)

            # Back Propagation
            layer3_error = t_outputs - layer3
            layer3_delta = layer3_error * self.tanh_prime(layer3) * self.learning_rate
            bias_3 += np.sum(layer3_delta)
            
            layer2_error = layer3_delta.dot(self.synapse_2.T)
            layer2_delta = layer2_error * self.tanh_prime(layer2) * self.learning_rate
            bias_2 += np.sum(layer2_delta)

            layer1_error = layer2_delta.dot(self.synapse_1.T)
            layer1_delta = layer1_error * self.tanh_prime(layer1) * self.learning_rate
            bias_1 += np.sum(layer1_delta)

            self.synapse_2 += layer2.T.dot(layer3_delta)
            self.synapse_1 += layer1.T.dot(layer2_delta)
            self.synapse_0 += layer0.T.dot(layer1_delta)
        
        self.training_time = time.time() - start_time

        if debug:
            print("Weight 0 = " + self.synapse_2)
            print("Weight 1 = " + self.synapse_1)
            print("Weight 2 = " + self.synapse_0)
            print('\n')
            print("Bias 1 = " + bias_1)
            print("Bias 2 = " + bias_2)
            print("Bias 3 = " + bias_3)

    def evolved_train(self, t_inputs, t_outputs, precision_percentage):
        start_time = time.time()
        
        bias_1 = bias_2 = bias_3 = 0

        delta = 0.0
        stagnation_iter = 0

        error = 1 - (precision_percentage/100.0)

        while(1):

            self.epoch += 1

            # Forward Propagation
            input_layer = t_inputs
            layer0 = input_layer
            layer1 = self.tanh(np.dot(layer0, self.synapse_0) + bias_1)
            layer2 = self.tanh(np.dot(layer1, self.synapse_1) + bias_2)
            layer3 = self.tanh(np.dot(layer2, self.synapse_2) + bias_3)
            

            # Back Propagation
            layer3_error = t_outputs - layer3
            layer3_delta = layer3_error * self.tanh_prime(layer3) * self.learning_rate
            bias_3 += np.sum(layer3_delta)
            
            layer2_error = layer3_delta.dot(self.synapse_2.T)
            layer2_delta = layer2_error * self.tanh_prime(layer2) * self.learning_rate
            bias_2 += np.sum(layer2_delta)

            layer1_error = layer2_delta.dot(self.synapse_1.T)
            layer1_delta = layer1_error * self.tanh_prime(layer1) * self.learning_rate
            bias_1 += np.sum(layer1_delta)

            self.synapse_2 += layer2.T.dot(layer3_delta)
            self.synapse_1 += layer1.T.dot(layer2_delta)
            self.synapse_0 += layer0.T.dot(layer1_delta)

            delta = layer3 - delta
            if delta.all() == 0.0:
                stagnation_iter += 1
            else:
                stagnation_iter = 0
            
            if stagnation_iter > 10000:
                print("Neural Network's training hasn't changed the output for 10000 iterations..! Stop training.")
                break

            if (layer1_error.any() < error and layer2_error.any() < error and layer3_error.any() < error):
                print("Training reached 97% accuracy and is complete.")
                break

            if self.epoch > 100000:
                break
        
        self.training_time = time.time() - start_time
