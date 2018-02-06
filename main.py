from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from layers import Standout
from utils import saveLog

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.99)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--standout', action='store_true', default=False,
                    help='Activates standout training!')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self, standout):
        super(Net, self).__init__()
        #### SELF ARGS ####
        self.standout = standout

        #### MODEL PARAMS ####
        self.fc1 = nn.Linear(784, 1000)
        self.fc1_drop = Standout(self.fc1, 0.5, 1) if standout else nn.Dropout(0.5)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc2_drop = Standout(self.fc2, 0.5, 1) if standout else nn.Dropout(0.5)
        self.fc_final = nn.Linear(1000, 10)

    def forward(self, x):
        # Flatten input
        x = x.view(-1, 784)
        # Keep it for standout

        #FIRST FC
        previous = x
        x_relu = F.relu(self.fc1(x))
        # Select between dropouts styles
        x = self.fc1_drop(previous, x_relu) if self.standout else self.fc1_drop(x_relu)

        #SECOND FC
        previous = x
        x_relu = F.relu(self.fc2(x))
        # Select between dropouts styles
        x = self.fc2_drop(previous, x_relu) if self.standout else self.fc2_drop(x_relu)

        x = self.fc_final(x)

        return F.log_softmax(x, dim=1)


def train(model, epoch):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test(model, standout, epoch):
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.5f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        test_acc))
    if standout == True:
        drop_way = "Standout"
    else:
        drop_way = "Dropout"
    saveLog(test_loss, test_acc, correct, drop_way, args, epoch)


def run(standout=False):

    model = Net(standout)
    if torch.cuda.is_available():
        model.cuda()

    test(model, standout, 0)
    for epoch in range(1, args.epochs + 1):
        train(model, epoch)
        test(model, standout, epoch)

def main():
    print("RUNNING STANDOUT ONE")
    run(standout=True)

    print("RUNNING DROPOUT ONE")
    run()

if __name__ == "__main__":
    main()
