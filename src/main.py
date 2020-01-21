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


def or_gate(epochs, in1, in2):
    # Or Gate prediction
    print("You're using the OR gate Neural Network...")

    t_inputs = np.array([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 1]])

    t_outputs = np.array([[0, 1, 1, 1]]).T

    or_predictor = NeuralNetwork()
    print("Training...")
    or_predictor.train(t_inputs, t_outputs, epochs)

    print("Training finished, predicting your outputs : ")

    predicted_output = or_predictor.predict(np.array([in1, in2]))
    
    print(np.array([in1, in2]), ' ---> ', np.round(predicted_output))
    

def and_gate(epochs, in1, in2):
    # And Gate prediction
    print("Training the AND gate model...")

def xor_gate(epochs, in1, in2):
    # Xor Gate prediction
    print("Training the XOR gate model...")

def not_gate(epochs, in1, in2):
    # Not Gate prediction
    print("Training the NOT gate model...")

def nand_gate(epochs, in1, in2):
    # Nand Gate prediction
    print("Training the NAND gate model...")

if __name__ == "__main__":
    ##### Getting args ######
    args = dict([arg.split('=') for arg in sys.argv[1:]])

    gate = args['gate']
    in1 = args['in1']
    in2 = args['in2']

    ##### Start Training #####
    or_gate(100, in1, in2)