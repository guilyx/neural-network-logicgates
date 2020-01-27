import numpy as np
from lib.network import Perceptron
from lib.network import NeuralNetwork

class LogicGate():
    def __init__(self, gate, activation = 'tanh', in1 = None, in2 = None):
        self.gate = gate

        if in1 == None:
            pass
        else:
            self.in1 = in1

        if in2 == None:
            pass
        else:
            self.in2 = in2

        self.activation = activation


    def print_gate(self):
        print("You're using the " + self.gate.upper() + " Neural Network")


    def choose_gate(self):
        if self.gate == 'or' or self.gate == 'OR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            self.t_outputs = np.array([[0, 1, 1, 1]]).T
        
        elif self.gate == 'xor' or self.gate == 'XOR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [1, 0],
                            [0, 1],
                            [1, 1]])

            self.t_outputs = np.array([[0, 1, 1, 0]]).T
        
        elif self.gate == 'nor' or self.gate == 'NOR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            self.t_outputs = np.array([[1, 0, 0, 0]]).T

        elif self.gate == 'and' or self.gate == 'AND':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            self.t_outputs = np.array([[0, 0, 0, 1]]).T
        
        elif self.gate == 'nand' or self.gate == 'NAND':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            self.t_outputs = np.array([[1, 1, 1, 0]]).T

        else:
            print("Error : Gate name not found !")

        return t_inputs, self.t_outputs

    def predict_output(self, n_epochs, learning_rate, mode):
        # Gate Prediction
        t_inputs, t_outputs = self.choose_gate()

        if (mode == 'perceptron' or mode == 'Perceptron' or mode == 'PERCEPTRON'):
            self.madame_irma = Perceptron(self.activation, learning_rate)
        elif (mode == 'network' or mode == 'Network' or mode == 'NETWORK'):
            self.madame_irma = NeuralNetwork(self.activation, learning_rate)
        print("Training...")
        self.madame_irma.train(t_inputs, t_outputs, n_epochs)

        training_time = "{0:.2f}".format(self.madame_irma.training_time) # rounding to two digits only
        print("Training finished in " + training_time + " seconds, predicting your output(s) : ")
        
        predicted_output = self.madame_irma.predict(np.array([self.in1, self.in2]))
        
        print(np.array([self.in1, self.in2]), ' ---> ', np.round(predicted_output))
        return(predicted_output)
    
    def evolved_predict_output(self, precision_percentage, learning_rate):
        # Gate Prediction
        t_inputs, t_outputs = self.choose_gate()
        
        self.madame_irma = NeuralNetwork(self.activation, learning_rate)

        print("Training...")
        self.madame_irma.evolved_train(t_inputs, t_outputs, precision_percentage)

        training_time = "{0:.2f}".format(self.madame_irma.training_time) # rounding to two digits only
        print("Training finished in " + training_time + " seconds, predicting your output(s) : ")
        
        predicted_output = self.madame_irma.evolved_predict(np.array([self.in1, self.in2]))
        
        print(np.array([self.in1, self.in2]), ' ---> ', np.round(predicted_output))
        return(predicted_output)


    def train(self, n_epochs, learning_rate, mode):
        # Gate Prediction
        t_inputs, t_outputs = self.choose_gate()

        if (mode == 'perceptron' or mode == 'Perceptron' or mode == 'PERCEPTRON'):
            self.madame_irma = Perceptron(self.activation, learning_rate)
        elif (mode == 'network' or mode == 'Network' or mode == 'NETWORK'):
            self.madame_irma = NeuralNetwork(self.activation, learning_rate)
        print("Training...")
        self.madame_irma.train(t_inputs, t_outputs, n_epochs)

        training_time = "{0:.2f}".format(self.madame_irma.training_time) # rounding to two digits only
        print("Training finished in " + training_time + " seconds, predicting your output(s) : ")


    def evolved_train(self, precision_percentage, learning_rate):
        # Gate Prediction
        t_inputs, t_outputs = self.choose_gate()
        
        self.madame_irma = NeuralNetwork(self.activation, learning_rate)

        print("Training...")
        self.madame_irma.evolved_train(t_inputs, t_outputs, precision_percentage)

        training_time = "{0:.2f}".format(self.madame_irma.training_time) # rounding to two digits only
        print("Training finished in " + training_time + " seconds, predicting your output(s) : ")


    def training_results(self):
        t_inputs, t_outputs = self.choose_gate()

        for elem in t_inputs:
            predicted_output = self.madame_irma.predict(elem)
            print(elem, ' ---> ', predicted_output)

        return(predicted_output)
    

    def evolved_training_results(self):
        t_inputs, t_outputs = self.choose_gate()

        for elem in t_inputs:
            predicted_output = self.madame_irma.evolved_predict(elem)
            print(elem, ' ---> ', predicted_output)

        return(predicted_output)