import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from common import mkdir, noise_inject
from AE.CVAE_GW import CVAE_GW, loss_for_vae
import MyNetModule as mnm



if __name__ == '__main__':


    filedir = 'testset/'

    trainwave = np.load(filedir+'dampedsinusoid_train.npy')
    trainlabel = torch.Tensor(np.genfromtxt(filedir+'dampedsinusoid_trainLabel.dat'))

    valwave = np.load(filedir+'dampedsinusoid_validate.npy')
    vallabel = torch.Tensor(np.genfromtxt(filedir+'dampedsinusoid_validateLabel.dat'))
    #valdata = torch.utils.data.TensorDataset(valwave, vallabel)


    hd = 128
    L = 1000
    Nepoch = 400
    alpha = 1.0
    batch_size = 32

    cvae = CVAE_GW(in_features=8192,
                   hidden_features=hd,
                   phys_features=2,
                   L = L)
    cvae.cuda()


    criterion = loss_for_vae(alpha=alpha)
    optimizer = optim.Adam(cvae.parameters())

    modeldir = '181127_01CVAEmodel/'
    mkdir(modeldir)
    figdir = modeldir+'figure/'
    mkdir(figdir)

    with open(modeldir+'model_property.txt', 'a+') as f:
        f.write('hidden features: %d\n' % hd)
        f.write('the sampling number of z: %d\n' % L)
        f.write('peak SNR: 100\n')
        f.write('total epoch: %d\n' % Nepoch)
        f.write('alpha in Adam: %.4f\n' % alpha)
        f.write('batch size: %d\n' % batch_size)



    pSNR = 1.0
    for epoch in range(Nepoch):

        #-----------------------------------------------------------
        # train and validate the model
        # START

        
        trainsignal = torch.Tensor(noise_inject(trainwave, pSNR))
        traindata = torch.utils.data.TensorDataset(trainsignal, trainlabel)
        data_loader = torch.utils.data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=8)

        valsignal = torch.Tensor(noise_inject(valwave, pSNR))

        cvae, running_loss = mnm.train_model(cvae, criterion, optimizer, data_loader, modeldir, epoch = epoch, modelsaveepoch=100)

        val_loss = mnm.validate_model(cvae, criterion, valsignal, vallabel)
        print("[%2d] %.5f %.5f" % (epoch+1, running_loss, val_loss.data))

        # END
        #-----------------------------------------------------------

        

        #-----------------------------------------------------------
        # check the model for 1 event (every 100 epoch)
        # START
        
        if epoch % 100 == 99:
            with torch.no_grad():

                j = 4625
                Nsample = 500000
                inputs, labels = traindata[j]
                inputs = inputs.view(1, 1, 8192)
                inputs = inputs.cuda()
                outlist = cvae.inference(inputs, Nsample)
                outlist = np.array(outlist).reshape(Nsample, 2)

            labels = labels.cpu().detach().numpy()

            bins = [np.linspace(100, 500, 401), np.linspace(5, 50, 46)]

            figname = figdir + 'fr%.3f_fi%.3f_%03d.png' % (labels[0], labels[1], epoch+1)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            counts, xedges, yedges, Image = ax.hist2d(outlist[:,0], outlist[:,1], bins=bins)    
            #ax.contour((counts / Nsample).transpose(),
            #           extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()])
            ax.plot(labels[0], labels[1], 'w+', markersize=10)
            ax.set_xlabel('frequency [Hz]')
            ax.set_ylabel('damping frequency [Hz]')
            fig.colorbar(Image, ax=ax)
            plt.savefig(figname)

            
        # END
        #-----------------------------------------------------------
        
    '''
    with torch.no_grad():
        for j in [0, 4625, 9245]:
            
            inputs, labels = traindata[j]
            inputs = inputs.view(1, 1, 8192)
            inputs = inputs.cuda()
            outputs, mu, Sigma = cvae(inputs)

            print(labels, outputs)



    with torch.no_grad():

        j = 4625
        Nsample = 100000
        inputs, labels = traindata[j]
        inputs = inputs.view(1, 1, 8192)
        inputs = inputs.cuda()
        outlist = cvae.inference(inputs, Nsample)


    outlist = np.array(outlist).reshape(Nsample, 2)

    bins = [np.linspace(70, 130, 61), np.linspace(30, 70, 41)]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    counts, xedges, yedges, Image = ax.hist2d(outlist[:,0], outlist[:,1], bins=bins)    
    #ax.contour((counts / Nsample).transpose(),
    #           extent=[xedges.min(),xedges.max(),yedges.min(),yedges.max()])
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('damping frequency [Hz]')
    fig.colorbar(Image, ax=ax)
    plt.savefig(figname)
    '''
