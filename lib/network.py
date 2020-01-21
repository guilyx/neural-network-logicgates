import numpy as np 

class NeuralNetwork():
    def __init__(self, synapses = None):

        if (synapses == None):
            self.synapses = 2 * np.random.random((2, 1)) - 1
        else: 
            self.synapses = synapses
