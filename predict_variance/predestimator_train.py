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
import MyNetModule as mnm
        

parser = argparse.ArgumentParser(description="parse args")
#parser.add_argument('-n', '--num-epochs', default=5, type=int)
#parser.add_argument('--datadir', default='/home/', type=str)
parser.add_argument('--savedir', type=str)
#parser.add_argument('--d', type=float)
args = parser.parse_args()


modeldir = '190220/'
savedir = args.savedir
mkdir(savedir)

net = mnm.DanielDeeperNet(out_features=2)
net.load_state_dict(torch.load(modeldir+'check_state.pth'))
net.cuda()

criterion = mnm.MeanRelativeError()
optimizer = optim.Adam(net.parameters(), lr=1e-4, weight_decay=0.001, amsgrad=True)


datadir = os.environ['HOME']+'/ringdown_net/'
trainwave = np.load(datadir+'TrainEOB_hPlus.npy')
trainlabel = torch.tensor(np.genfromtxt(datadir+'TrainLabel_new.dat')[:,0:2], dtype=torch.float32)
validatewave = np.load(datadir+'ValidateEOB_hPlus.npy')
validatelabel = torch.tensor(np.genfromtxt(datadir+'ValidateLabel_new.dat')[:,0:2], dtype=torch.float32)



snr_list = [0.6]

tik = time.time()
for snr_loc in snr_list:

    pSNR = [snr_loc, 4.0]
    
    if snr_loc>3.5: nepoch = 3
    elif snr_loc>2.0: nepoch = 3
    else: nepoch = 10
    
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
            preds = net(inputs)
            loss = criterion(preds, labels)

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

                preds = net(inputs)
                val_loss += criterion(preds, labels).data


        with open(savedir+'log4check_runningloss_pred.txt', 'a') as f:
            f.write("%.1f %.d %.5f %.5f\n" % (snr_loc, epoch+1, running_loss/(i+1), val_loss.data/(j+1)))


torch.save(net.state_dict(), savedir+'check_state_prednet.pth')
torch.save(optimizer.state_dict(), savedir+'check_optim_prednet.pth')

tok = time.time()


print("elapsed time:{0} [sec]".format(tok-tik))

