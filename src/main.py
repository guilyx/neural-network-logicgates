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

if __name__ == "__main__":
    ##### Getting args ######
    args = dict([arg.split('=') for arg in sys.argv[1:]])

    gate = args['gate']
    in1 = args['in1']
    in2 = args['in2']

    epoch = 10000

    ##### Start Training #####
    predict_gate(epoch, in1, in2, gate)