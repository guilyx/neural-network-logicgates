'''
@author : Erwin Lejeune <erwin.lejeune15@gmail.com>
@date : 21/01/2020
@brief : convolutional neural network predicting the results 
of inputs depending on the gate used

@args : gate=and/or/xor/nand/not in1=0 in2=1
'''

import numpy as np
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from lib.network import NeuralNetwork

def print_gate(gate):
    print("You're using the " + gate.upper() + " Neural Network")

def choose_gate(gate):
    if gate == 'or' or gate == 'OR':
        print_gate(gate)

        t_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

        t_outputs = np.array([[0, 1, 1, 1]]).T
    
    elif gate == 'xor' or gate == 'XOR':
        print_gate(gate)

        t_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

        t_outputs = np.array([[0, 1, 1, 0]]).T
    
    elif gate == 'nor' or gate == 'NOR':
        print_gate(gate)

        t_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

        t_outputs = np.array([[1, 0, 0, 0]]).T

    elif gate == 'and' or gate == 'AND':
        print_gate(gate)

        t_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

        t_outputs = np.array([[0, 0, 0, 1]]).T
    
    elif gate == 'nand' or gate == 'NAND':
        print_gate(gate)

        t_inputs = np.array([[0, 0],
                        [0, 1],
                        [1, 0],
                        [1, 1]])

        t_outputs = np.array([[1, 1, 1, 0]]).T

    else:
        print("Error : gate name not found !")

    return t_inputs, t_outputs

def predict_gate(epochs, in1, in2, gate):
    # Or Gate prediction
    t_inputs, t_outputs = choose_gate(gate)

    madame_irma = NeuralNetwork()
    print("Training...")
    madame_irma.train(t_inputs, t_outputs, epochs)

    training_time = "{0:.2f}".format(madame_irma.training_time) # rounding to two digits only
    print("Training finished in " + training_time + " seconds, predicting your output(s) : ")

    predicted_output = madame_irma.predict(np.array([in1, in2]))
    
    print(np.array([in1, in2]), ' ---> ', np.round(predicted_output))

if __name__ == "__main__":
    ##### Getting args ######
    args = dict([arg.split('=') for arg in sys.argv[1:]])

    gate = args['gate']
    in1 = args['in1']
    in2 = args['in2']

    epoch = 10000

    ##### Start Training #####
    predict_gate(epoch, in1, in2, gate)