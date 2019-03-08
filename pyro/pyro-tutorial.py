import torch
from torch.distributions import constraints

import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

def model(data):
    alpha0 = torch.tensor(10.0)
    beta0 = torch.tensor(10.0)

    #sample f from beta prior
    f = pyro.sample("latent_fairness", dist.Beta(alpha0, beta0))

    #loop over the observed data
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])


def guide(data):
    alpha_q = pyro.param("alpha_q", torch.tensor(15.0),
                         constraint=constraints.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.0),
                        constraint=constraints.positive)

    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))

if __name__=='__main__':

    adam_params = {"lr": 0.0005, "betas": (0.90, 0.999)}
    optimizer = pyro.optim.Adam(adam_params)

    svi = pyro.infer.SVI(model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

    n_steps = 1000

    for step in range(n_steps):
        svi.step(data)


