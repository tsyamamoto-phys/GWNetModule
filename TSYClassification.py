import torch

def vector2binaryclass(vec, threshold=0.5):

    """ NN's output vector is converted into a label of class.

    parameters:
    vec: torch.tensor float vecor
    vec[:,0] = probability of null hypothesis
    vec[:,1] = probability of alternative hypothesis
    threshold (default=0.5): detection threshold

    returns:
    classint: torch.long
    """
    
    return ((x > threshold)[:,1]).type(torch.long)

