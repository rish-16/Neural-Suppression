import torch
import torch.nn
import torch.nn.functional as F
from torch.distributions.binomial import Binomial

class Suppress2d(nn.Module):
    def __init__(self, gamma=0.2, p=0.5):
        super().__init__()
        self.p = p # probability of not discounting a neuron (ie retaining a neuron)
        self.gamma = gamma # discount factor
        binomial = Binomial(probs=q)

    def forward(self, x):
        '''
        - Randomly select neurons
        - Apply discount to selected neurons
        - Propagate discounted and undiscounted values to next layer

        X   M   M'    g       COMBINE       Out
        a * 1 = a  * 0.2 = 0.2a + (a * 0) = 0.2a
        b * 1 = b  * 0.2 = 0.2b + (b * 0) = 0.2b
        c * 0 = 0  * 0.2 =   0  + (c * 1) = c
        d * 1 = d  * 0.2 = 0.2d + (d * 0) = 0.2d
        e * 0 = 0  * 0.2 =   0  + (e * 0) = e
        '''

        if not self.training: # model in eval mode
            return x

        q = 1 - self.p # probability of neurons being discounted
        
        mask = self.binomial.sample(x.size()) # get the neurons to be discount
        discounted = x * mask * self.gamma
        
        inverted_mask = torch.logical_not(mask).float() # get the inverted mask
        remnant = x * inverted_mask * 1/q

        return discounted + remnant
