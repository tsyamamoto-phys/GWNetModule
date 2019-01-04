import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def inputs_reshape(array, Lin):

    """
    array should be torch.Tensor
    """
    Nb, _, Lseq = array.size()
    array_reshaped = torch.empty([Nb, Lseq-Lin+1, Lin])

    for n in range(Lseq-Lin+1):
        array_reshaped[:, n, :] = array[:, 0, n:n+Lin]

    return array_reshaped



class RNN_for_Ringdown(nn.Module):

    def __init__(self, Lseq, Lin, Lhidden):
        super(RNN_for_Ringdown, self).__init__()

        self.Lseq = Lseq
        self.Lin = Lin
        self.Lhidden = Lhidden
        
        self.rnn = nn.RNN(input_size = Lin,
                          hidden_size = Lhidden,
                          num_layers = 2,
                          nonlinearity = 'relu',
                          batch_first = True,
                          bidirectional = True)

        self.fc = nn.Linear(in_features = Lhidden*2 * Lseq,
                            out_features = 2)


    def forward(self, inputs):

        z, h = self.rnn(inputs)
        print(z.size())
        z = z.contiguous().view(-1, self.Lhidden * self.Lseq * 2)
        return self.fc(z)



if __name__=='__main__':

    a = torch.rand(5,1,100)
    Lin = 10
    Lseq = 100 - Lin + 1

    b = inputs_reshape(a, Lin)

    print(a[0,0,0:20])
    print(b[0,1,:])
    
    print(b.size())


    model = RNN_for_Ringdown(Lseq = Lseq,
                             Lin = Lin,
                             Lhidden = 16)

    outputs = model(b)
    print(outputs.size())
