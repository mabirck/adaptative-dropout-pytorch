import torch
from torch.autograd import Variable
from torch import nn

class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        print("<<<<<<<<< THIS IS DEFINETLY A STANDOUT TRAINING >>>>>>>>>>>>>>>")
        super(Standout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()


    def forward(self, previous, current, p=0.5, deterministic=False):
        # Function as in page 3 of paper: Variational Dropout
        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        # Deterministic version as in the paper
        if(deterministic or torch.mean(self.p).data.cpu().numpy()==0):
            return self.p * current
        else:
            return self.mask * current

def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch."""

    if torch.cuda.is_available():
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1).cuda())
    else:
        uniform = Variable(torch.Tensor(p.size()).uniform_(0, 1))
    mask = uniform < p

    if torch.cuda.is_available():
        mask = mask.type(torch.cuda.FloatTensor)
    else:
        mask = mask.type(torch.FloatTensor)

    return mask
