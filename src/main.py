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
from lib.gate_predictions import LogicGate

if __name__ == "__main__":
    ##### Getting args ######
    args = dict([arg.split('=') for arg in sys.argv[1:]])

    gate = args['gate']
    in1 = args['in1']
    in2 = args['in2']

    epoch = 10000
    learning_rate = 0.05

    ##### Start Training #####
    gate_ = LogicGate(gate, in1, in2)
    pr_output = gate_.predict_gate(epoch, learning_rate)

    print(pr_output)