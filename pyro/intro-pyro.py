import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms

import pyro
from pyro.distributions import Normal
from pyro.distributions import Categorical
from pyro.optim import Adam
from pyro.infer import SVI
from pyro.infer import Trace_ELBO

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()



class BNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(BNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=16)
        self.pool1 = nn.MaxPooling1d(4, stride=4)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=8, dilation=4)
        self.pool2 = nn.MaxPooling1d(4, stride=4)
        self.conv3= nn.Conv1d(in_channels=32, out_channels=64, kernel_size=8, dilation=4)
        self.pool3 = nn.MaxPooling1d(4, stride=4)
        self.fc1 = nn.Linear(64*119, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(-1, 64*119)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))
        fc1b_prior = Normal(loc=torch.zeros_like(self.fc1.bias),
                            scale=torch.ones_like(self.fc1.bias))
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))
        fc1w_prior = Normal(loc=torch.zeros_like(self.fc1.weight),
                            scale=torch.ones_like(self.fc1.weight))


net = BNN(1024, 10)
net.cuda()

def model(x_data, y_data):
    # define prior destributions

    fc1w_prior = Normal(loc=torch.zeros_like(net.fc1.weights),
                        scale=torch.ones_like(net.fc1.weights))
    fc1b_prior = Normal(loc=torch.zeros_like(net.fc1.bias),
                        scale=torch.ones_like(net.fc1.bias))
    fc2w_prior = Normal(loc=torch.zeros_like(net.fc2.weights),
                        scale=torch.ones_like(net.fc2.weights))
    fc2b_prior = Normal(loc=torch.zeros_like(net.fc2.bias),
                        scale=torch.ones_like(net.fc2.bias))

    priors = {'fc.1weight': fc1w_prior,
              'fc1.bias': fc1b_prior, 
              'out.weight': outw_prior,
              'out.bias': outb_prior}

    
    lifted_module = pyro.random_module("module", net, priors)
    lifted_reg_model = lifted_module()
    
    yhat = lifted_reg_model(x_data)
    pyro.sample("obs", Categorical(logits=lhat), obs=y_data)



def guide(x_data, y_data):
    fc1w_mu = torch.randn_like(net.fc1.weight)
    fc1w_sigma = torch.randn_like(net.fc1.weight)
    fc1w_mu_param = pyro.param("fc1w_mu", fc1w_mu)
    fc1w_sigma_param = F.softplus(pyro.param("fc1w_sigma", fc1w_sigma))
    fc1w_prior = Normal(loc=fc1w_mu_param, scale=fc1w_sigma_param)

    fc1b_mu = torch.randn_like(net.fc1.bias)
    fc1b_sigma = torch.randn_like(net.fc1.bias)
    fc1b_mu_param = pyro.param("fc1b_mu", fc1b_mu)
    fc1b_sigma_param = F.softplus(pyro.param("fc1b_sigma", fc1b_sigma))
    fc1b_prior = Normal(loc=fc1b_mu_param, scale=fc1b_sigma_param)

    outw_mu = torch.randn_like(net.out.weight)
    outw_sigma = torch.randn_like(net.out.weight)
    outw_mu_param = pyro.param("outw_mu", outw_mu)
    outw_sigma_param = F.softplus(pyro.param("outw_sigma", outw_sigma))
    outw_prior = Normal(loc=outw_mu_param, scale=outw_sigma_param).independent(1)

    outb_mu = torch.randn_like(net.out.bias)
    outb_sigma = torch.randn_like(net.out.bias)
    outb_mu_param = pyro.param("outb_mu", outb_mu)
    outb_sigma_param = F.softplus(pyro.param("outb_sigma", outb_sigma))
    outb_prior = Normal(loc=outb_mu_param, scale=outb_sigma_param)
    
    priors = {
        'fc1.weight': fc1w_prior,
        'fc1.bias': fc1b_prior,
        'out.weight': outw_prior,
        'out.bias': outb_prior}
    
    lifted_module = pyro.random_module("module", net, priors)
    
    return lifted_module()

optim = Adam({"lr": 0.01})
svi = SVI(model, guide, optim, loss=Trace_ELBO())

n_iterations = 5
loss = 0

for j in range(n_iterations):
    loss = 0
    for batch_id, data in enumerate(train_loader):
        data[0] = data[0].cuda()
        data[1] = data[1].cuda()
        loss += svi.step(data[0].view(-1,1,28,28), data[1])

    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = loss / normalizer_train
    
    print("Epoch ", j, " Loss ", total_epoch_loss_train)


net.cpu()

n_samples = 10

def predict(x):
    sampled_models = [guide(None, None) for _ in range(n_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return np.argmax(mean.cpu().numpy(), axis=1)

print('Prediction when network is forced to predict')
"""
correct = 0
total = 0
for j, data in enumerate(test_loader):
    images, labels = data
    images = images.cuda()
    labels = labels.cuda()
    predicted = predict(images.view(-1,1,28,28))
    total += labels.size(0)
    correct += (predicted == labels.cpu().numpy()).sum().item()
print("accuracy: %d %%" % (100 * correct / total))
"""
def predict_prob(x):
    sampled_models = [guide(None, None) for _ in range(n_samples)]
    yhats = [model(x).data for model in sampled_models]
    mean = torch.mean(torch.stack(yhats), 0)
    return mean

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def plot(x, yhats):
    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(8, 4))
    axL.bar(x=[i for i in range(10)], height= F.softmax(torch.Tensor(normalize(yhats.cpu().numpy()))[0]))
    axL.set_xticks([i for i in range(10)], [i for i in range(10)])
    axR.imshow(x.cpu().numpy()[0])
    plt.show()

x, y = test_loader.dataset[0]
x = x.cuda()
y = y.cuda()
yhats = predict_prob(x.view(-1,1,28,28))
print("ground truth: ", y.cpu().item())
print("predicted: ", yhats.cpu().numpy())
plot(x, yhats)
