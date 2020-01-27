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

from lib.network import Perceptron
from lib.network import NeuralNetwork
from lib.gate_predictions import LogicGate

def main():
    ##### Getting args ######
    args = dict([arg.split('=') for arg in sys.argv[1:]])

    gate = args['gate']
    in1 = args['in1']
    in2 = args['in2']
    mode = args['mode']

    if mode == 'perceptron':
        epoch = 10000
    elif mode == 'network':
        epoch = 1000000

    learning_rate = 1

    ##### Start Training #####
    gate_ = LogicGate(gate, in1, in2)
    pr_output = gate_.predict_output(epoch, learning_rate, mode)

    print(pr_output)

def gate_outputs(activation):
    args = dict([arg.split('=') for arg in sys.argv[1:]])
    gate = args['gate']
    mode = args['mode']

    learning_rate = .1
    evolved = True

    gate_ = LogicGate(gate, activation)

    if mode == 'perceptron':
        epoch = 100000
        gate_.train(epoch, learning_rate, mode)
    elif mode == 'network':
        epoch = 100000
        if evolved:
            accuracy = 95
            gate_.evolved_train(accuracy, learning_rate)
            pr_outputs = gate_.evolved_training_results()
        else:
            gate_.train(epoch, learning_rate, mode)
            pr_outputs = gate_.training_results()
    


if __name__ == "__main__":
    gate_outputs('sigmoid')
    gate_outputs('tanh')