import numpy as np
from lib.network import NeuralNetwork

class LogicGate():
    def __init__(self, gate, in1, in2):
        self.gate = gate
        self.in1 = in1
        self.in2 = in2


    def print_gate(self):
        print("You're using the " + self.gate.upper() + " Neural Network")


    def choose_gate(self):
        if self.gate == 'or' or self.gate == 'OR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            t_outputs = np.array([[0, 1, 1, 1]]).T
        
        elif self.gate == 'xor' or self.gate == 'XOR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            t_outputs = np.array([[0, 1, 1, 0]]).T
        
        elif self.gate == 'nor' or self.gate == 'NOR':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            t_outputs = np.array([[1, 0, 0, 0]]).T

        elif self.gate == 'and' or self.gate == 'AND':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            t_outputs = np.array([[0, 0, 0, 1]]).T
        
        elif self.gate == 'nand' or self.gate == 'NAND':
            self.print_gate()

            t_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

            t_outputs = np.array([[1, 1, 1, 0]]).T

        else:
            print("Error : self.gate name not found !")

        return t_inputs, t_outputs

    def predict_gate(self, epochs):
        # Gate Prediction
        t_inputs, t_outputs = self.choose_gate()

        madame_irma = NeuralNetwork()
        print("Training...")
        madame_irma.train(t_inputs, t_outputs, epochs)

        training_time = "{0:.2f}".format(madame_irma.training_time) # rounding to two digits only
        print("Training finished in " + training_time + " seconds, predicting your output(s) : ")

        predicted_output = madame_irma.predict(np.array([self.in1, self.in2]))
        
        print(np.array([self.in1, self.in2]), ' ---> ', np.round(predicted_output))