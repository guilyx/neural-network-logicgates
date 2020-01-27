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
    learning_rate = 0.1
    error = .25
    epochs = 100000
    mode = 'perceptron'

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 1, 1, 1]]).T
    
    gate_ = LogicGate(gate, 'sigmoid')
    gate_.train(epochs, learning_rate, mode)

    i = 0

    for elem in expected_inputs:
        pr_out = gate_.madame_irma.predict(elem)
        print(elem, ' ---> ', pr_out[0])
        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)
        i += 1


def test_outputs_nor():
    gate = 'nor'
    learning_rate = 0.1
    error = .25
    epochs = 100000
    mode = 'perceptron'

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[1, 0, 0, 0]]).T
    
    gate_ = LogicGate(gate, 'sigmoid')
    gate_.train(epochs, learning_rate, mode)

    i = 0

    for elem in expected_inputs:
        pr_out = gate_.madame_irma.predict(elem)
        print(elem, ' ---> ', pr_out[0])
        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)
        i += 1


def test_outputs_xor():
    gate = 'xor'
    learning_rate = 0.1
    error = .25
    epochs = 100000
    mode = 'perceptron'

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 1, 1, 0]]).T
    
    gate_ = LogicGate(gate, 'sigmoid')
    gate_.train(epochs, learning_rate, mode)

    i = 0

    for elem in expected_inputs:
        pr_out = gate_.madame_irma.predict(elem)
        print(elem, ' ---> ', pr_out[0])
        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)
        i += 1


def test_outputs_and():
    gate = 'and'
    learning_rate = 0.1
    error = .25
    epochs = 100000
    mode = 'perceptron'

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[0, 0, 0, 1]]).T
    
    gate_ = LogicGate(gate, 'sigmoid')
    gate_.train(epochs, learning_rate, mode)

    i = 0

    for elem in expected_inputs:
        pr_out = gate_.madame_irma.predict(elem)
        print(elem, ' ---> ', pr_out[0])
        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)
        i += 1


def test_outputs_nand():
    gate = 'nand'
    learning_rate = 0.1
    error = .25
    epochs = 100000
    mode = 'perceptron'

    expected_inputs = np.array([[0, 0],
                         [0, 1],
                         [1, 0],
                         [1, 1]])

    expected_outputs = np.array([[1, 1, 1, 0]]).T
    
    gate_ = LogicGate(gate, 'sigmoid')
    gate_.train(epochs, learning_rate, mode)

    i = 0

    for elem in expected_inputs:
        pr_out = gate_.madame_irma.predict(elem)
        print(elem, ' ---> ', pr_out[0])
        assert(pr_out[0] > expected_outputs[i][0] - error and pr_out[0] < expected_outputs[i][0] + error)
        i += 1