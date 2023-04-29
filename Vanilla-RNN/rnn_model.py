'''
Vanilla-RNN implementation using PyTorch.
'''

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    '''Vanilla RNN model.
    '''


    def __init__(self, hidden_input_size, hidden_layer_size, input_layer_size, output_layer_size):
        '''Param init.
        '''
        super(VanillaRNN, self).__init__()

        self.W_xh = nn.Linear(input_layer_size, hidden_layer_size)
        self.W_h1 = nn.Linear(hidden_input_size, hidden_layer_size)
        self.W_hh = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.W_hy = nn.Linear(hidden_layer_size, output_layer_size)


        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, prev_hidden_state, first_run=False):
        '''Forward Propagation.
        '''

        input_to_hidden = self.W_xh(x)

        curr_hidden_state = None
        if first_run:
            curr_hidden_state = self.tanh(input_to_hidden + self.W_h1(prev_hidden_state))
        else:
            curr_hidden_state = self.tanh(input_to_hidden + self.W_hh(prev_hidden_state))

        curr_output = self.softmax(self.W_hy(curr_hidden_state))

        return curr_output, curr_hidden_state






