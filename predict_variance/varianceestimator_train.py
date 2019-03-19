import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import argparse
import time

import os, sys
sys.path.append(os.environ['HOME']+'/GWNetModule')
from common import noise_inject, mkdir
from errornet_module import ErrorEstimateNet_double, reducedNLL

        

parser = argparse.ArgumentParser(description="parse args")
#parser.add_argument('-n', '--num-epochs', default=5, type=int)
#parser.add_argument('--datadir', default='/home/', type=str)
parser.add_argument('--savedir', type=str)
#parser.add_argument('--d', type=float)
args = parser.parse_args()


modeldir = '190220/'
savedir = args.savedir
mkdir(savedir)

net = ErrorEstimateNet_double()

model_dict = net.state_dict()
pretrained_dict = torch.load(modeldir+'check_state.pth')
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)

net.freeze_pred_net()
net.cuda()

criterion = reducedNLL()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
'''
optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()),
        lr=1e-4, weight_decay=0.01, amsgrad=True)
'''

datadir = os.environ['HOME']+'/ringdown_net/'
trainwave = np.load(datadir+'TrainEOB_hPlus.npy')
trainlabel = torch.tensor(np.genfromtxt(datadir+'TrainLabel_new.dat')[:,0:2], dtype=torch.float32)
validatewave = np.load(datadir+'ValidateEOB_hPlus.npy')
validatelabel = torch.tensor(np.genfromtxt(datadir+'ValidateLabel_new.dat')[:,0:2], dtype=torch.float32)



#snr_list = np.arange(4.0, 0.5, -0.1)
snr_list = [4.0]

tik = time.time()
for snr_loc in snr_list:

    pSNR = snr_loc
    
    if snr_loc>3.5: nepoch = 50
    elif snr_loc>2.0: nepoch = 3
    else: nepoch = 3
    
    for epoch in range(nepoch):

        trainsignal, _ = noise_inject([trainwave], pSNR)
        trainsignal = torch.tensor(trainsignal, dtype=torch.float32)
        traindata = torch.utils.data.TensorDataset(trainsignal,trainlabel)
        train_loader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=4, drop_last=True)

        validatesignal, _ = noise_inject([validatewave], pSNR)
        validatesignal = torch.tensor(validatesignal, dtype=torch.float32)
        validatedata = torch.utils.data.TensorDataset(validatesignal, validatelabel)
        validate_loader = torch.utils.data.DataLoader(validatedata, batch_size=256, shuffle=True, num_workers=4)

        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            preds = net.point_pred(inputs)
            var = net.variance(inputs)
            loss = criterion(labels, preds, var)

            running_loss += loss.data
            loss.backward()
            optimizer.step()


            print("%02d %03d %.5f" % (epoch+1, i+1, loss.data))
           

        with torch.no_grad():
            val_loss = 0.0
            for j, data in enumerate(validate_loader, 0):
                
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                preds = net.point_pred(inputs)
                var = net.variance(inputs)
                val_loss += criterion(labels, preds, var).data


        with open(savedir+'log4check_runningloss_var.txt', 'a') as f:
            f.write("%.1f %.d %.5f %.5f\n" % (snr_loc, epoch+1, running_loss/(i+1), val_loss.data/(j+1)))


torch.save(net.state_dict(), savedir+'check_state_varnet.pth')
torch.save(optimizer.state_dict(), savedir+'check_optim_varnet.pth')

tok = time.time()


print("elapsed time:{0} [sec]".format(tok-tik))

