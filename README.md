# adaptative-dropout-pytorch
Pytorch implementation of Adaptive Dropout a.k.a Standout.

* Unfortunately I wasn't able to achieve the results reported in the paper, Regular dropout always get aproximately 98.70, while the standout version I was able to achieve only 98.51 after tweaking a lot the knoobs, i didn't make grid search was all empirical and cloning the paper parameters, furthermore I did not make any unsupervized pretraining, maybe this tecnique is very sensitive to this step! Hope someone can achieve the reporteded results from this code.

# References:

  -Papers:
    https://papers.nips.cc/paper/5032-adaptive-dropout-for-training-deep-neural-networks.pdf

  -Code:

    https://github.com/gngdb/adaptive-standout

    https://github.com/pytorch/examples
