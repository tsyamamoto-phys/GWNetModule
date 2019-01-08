import numpy as np
#import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def inputs_reshape(array, Lin):

    """
    array should be torch.Tensor
    """
    Nb, _, Lseq = array.size()
    padd = int(Lin / 2)
    array_reshaped = torch.zeros([Nb, Lseq, Lin])

    for n in range(Lseq):

        if n<padd:
            array_reshaped[:, n, padd-n:Lin] = array[:, 0, 0:Lin-padd+n]

        elif n>=Lseq-padd:
            array_reshaped[:, n, 0:Lseq-n+padd] = array[:, 0, n-padd:Lseq]

        else:
            array_reshaped[:, n, :] = array[:, 0, n-padd:n-padd+Lin]

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





class RNN_for_denoising(nn.Module):

    def __init__(self, Lseq, Lin, Lhidden):
        super(RNN_for_denoising, self).__init__()

        self.Lseq = Lseq
        self.Lin = Lin
        self.Lhidden = Lhidden
        
        self.lstm1 = nn.LSTM(input_size = Lin,
                             hidden_size = Lhidden,
                             num_layers = 1,
                             batch_first = True,
                             bidirectional = False)


        
        self.lstm2 = nn.LSTM(input_size = Lhidden,
                             hidden_size = 1,
                             num_layers = 1,
                             batch_first = True,
                             bidirectional = False)
        


    def forward(self, inputs):

        z, _ = self.lstm1(inputs)
        z, _ = self.lstm2(z)
        return z.view(-1, self.Lseq)





    

if __name__=='__main__':

    a = torch.rand(5,1,10)
    Lin = 5
    Lseq = 10

    b = inputs_reshape(a, Lin)

    print(a[0,0,0:10])
    print(b[0,0,:])
    print(b[0,1,:])
    print(b[0,2,:])
    
    print("input size", b.size())


    model = RNN_for_denoising(Lseq = Lseq,
                              Lin = Lin,
                              Lhidden = 16)

    outputs = model(b)
    print("output size", outputs.size())
