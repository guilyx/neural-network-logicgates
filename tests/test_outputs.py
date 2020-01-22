import numpy as np
from random import randint
from random import uniform
import sys
from os import path
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )

from lib.network import NeuralNetwork
from lib.gate_predictions import LogicGate

def test_outputs_or():
    gate = 'or'
    epochs = randint(100, 100000)
    learning_rate = 0.05
    
    # error = 10/float(epochs)
    error = .1

    error = .1

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 1, 1, 1]]).T
    
    for i in range(3):
        in1 = expected_inputs[i][0]
        in2 = expected_inputs[i][1]

        gate_ = LogicGate(gate, in1, in2)
        pr_out = gate_.predict_gate(epochs, learning_rate)

        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)

def test_outputs_nor():
    gate = 'nor'
    epochs = randint(100, 100000)
    learning_rate = 0.05
    
    # error = 10/float(epochs)
    error = .1

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[1, 0, 0, 0]]).T
    
    for i in range(3):
        in1 = expected_inputs[i][0]
        in2 = expected_inputs[i][1]

        gate_ = LogicGate(gate, in1, in2)
        pr_out = gate_.predict_gate(epochs, learning_rate)

        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)


def test_outputs_xor():
    gate = 'xor'
    epochs = randint(100, 100000)
    learning_rate = 0.05
    
    # error = 10/float(epochs)
    error = .1

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 1, 1, 0]]).T
    
    for i in range(3):
        in1 = expected_inputs[i][0]
        in2 = expected_inputs[i][1]

        gate_ = LogicGate(gate, in1, in2)
        pr_out = gate_.predict_gate(epochs, learning_rate)

        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)


def test_outputs_and():
    gate = 'and'
    epochs = randint(100, 100000)
    learning_rate = 0.05
    
    # error = 10/float(epochs)
    error = .1

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 0, 0, 1]]).T
    
    for i in range(3):
        in1 = expected_inputs[i][0]
        in2 = expected_inputs[i][1]

        gate_ = LogicGate(gate, in1, in2)
        pr_out = gate_.predict_gate(epochs, learning_rate)

        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)


def test_outputs_nand():
    gate = 'nand'
    epochs = randint(100, 100000)
    learning_rate = 0.05
    
    # error = 10/float(epochs)
    error = .1

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[1, 1, 1, 0]]).T
    
    for i in range(3):
        in1 = expected_inputs[i][0]
        in2 = expected_inputs[i][1]

        gate_ = LogicGate(gate, in1, in2)
        pr_out = gate_.predict_gate(epochs, learning_rate)

        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)