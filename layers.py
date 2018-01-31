import torch
from torch.autograd import Variable
from torch import nn

class Standout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        super(Standout, self).__init__()
        #print(dir(last_layer))
        #self.last_input = last_layer.output
        self.W = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        print(self.W.size())
        self.pi = self.alpha * self.W + self.beta
        self.nonlinearity = nn.Sigmoid()


    def forward(self, inputs):
        print(inputs.size(), self.pi.size())
        self.p = self.nonlinearity(torch.dot(inputs, self.pi))
        self.mask = sample_mask(self.p)

        if(deterministic or torch.mean(self.p) ==0):
            return self.p * inputs
        else:
            return self.mask * inputs

    def sample_mask(p):
        """Given a matrix of probabilities, this will sample a mask in PyTorch."""

        uniform = torch.Tensor(p.size()).uniform(0, 1)
        mask = uniform < pdb
        return mask

# class DropoutAlgorithm2(lasagne.layers.MergeLayer):
#     def nothing():
#         """
#         Algorithm 2 reuses parameters from the previous hidden layer to perform
#         the forward prop used for dropout probabilities. Then, these are scaled
#         with alpha and beta. To make this layer work, you must place a
#         DropoutCallForward layer before the layer before this dropout layer. When
#         calling lasagne.layers.get_output, the CallForward layer will ensure the
#         expression for p is using the correct input variable.
#         Inputs:
#             * incoming - previous layer
#             * alpha - scale hyperparameter
#             * beta - shift hyperparameter
#         """
#     def __init__(self, incoming, alpha, beta,
#             nonlinearity=lasagne.nonlinearities.sigmoid, **kwargs):
#         # pull the layer before the incoming layer out of chain
#         incoming_input = lasagne.layers.get_all_layers(incoming)[-2]
#         lasagne.layers.MergeLayer.__init__(self, [incoming, incoming_input], **kwargs)
#         self.W = incoming.W
#         self.alpha = alpha
#         self.beta = beta
#         self.pi = self.alpha*self.W + self.beta
#         self.nonlinearity = nonlinearity
#
#     def get_output_for(self, inputs, deterministic=False):
#         self.p = self.nonlinearity(T.dot(inputs[1], self.pi))
#         self.mask = sample_mask(self.p)
#         if deterministic or T.mean(self.p) == 0:
#             return self.p*inputs[0]
#         else:
#             return inputs[0]*self.mask
#
#     def get_output_shape_for(self, input_shapes):
#         """
#         Layer will always return the input shape of the incoming layer, because
#         it's just applying a mask to that layer.
#         """
#         return input_shapes[0]
#
# def sample_mask(p):
#     """
#     give a matrix of probabilities, this will sample a mask. Theano.
#     """
#     # sample uniform and threshold to sample from many different
#     # probabilities
#     uniform = _srng.uniform(p.shape)
#     # keep if less than retain_prob
#     mask = uniform < p
#     return mask
#
# def get_all_beliefnets(output_layer, input_var):
#     """
#     Takes an output layer and collects up all the belief net layers in the
#     network. Useful for gathering up the parameters so that they can then
#     be updated.
#     """
#     all_layers = lasagne.layers.get_all_layers(output_layer, input_var)
#     all_beliefnets = [l.incoming_beliefnet for l in all_layers
#                       if isinstance(l, Dropout)]
#     return all_beliefnets
#
# def get_all_parameters(output_layer):
#     """
#     Analogous to the lasagne.layers.get_all_parameters, but gathers only the
#     parameters of the standout layers.
#     """
#     all_beliefnets = get_all_beliefnets(output_layer)
#     return [l.W for l in all_beliefnets]
