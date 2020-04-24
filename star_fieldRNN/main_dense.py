import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import sys
sys.path.append("../../")
from TCN.mnist_pixel.utils import data_generator
from TCN.fieldRNN.model_dense import TCN
import numpy as np
import argparse
from dataloader import Dataloader


parser = argparse.ArgumentParser(description='Sequence Modeling - (Permuted) Sequential MNIST')
parser.add_argument('--batch_size', type=int, default=500, metavar='N',
                    help='batch size (default: 64)')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA (default: True)')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (default: 0.05)')
parser.add_argument('--clip', type=float, default=-1,
                    help='gradient clip, -1 means no clip (default: -1)')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit (default: 20)')
parser.add_argument('--ksize', type=int, default=7,
                    help='kernel size (default: 7)')
parser.add_argument('--levels', type=int, default=2,
                    help='# of levels (default: 8)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='report interval (default: 100')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate (default: 2e-3)')
parser.add_argument('--optim', type=str, default='Adam',
                    help='optimizer to use (default: Adam)')
parser.add_argument('--nhid', type=int, default=25,
                    help='number of hidden units per layer (default: 25)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed (default: 1111)')
parser.add_argument('--permute', action='store_true',
                    help='use permuted MNIST (default: false)')
args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

batch_size = 500
n_classes = 23
input_channels = 55
seq_length = int(24)
epochs = args.epochs
steps = 0
print(args)

train_localdir = "/home/pf/pfstaff/projects/ozgur_deep_filed/download/fieldRNN_data/train"
test_localdir = "/home/pf/pfstaff/projects/ozgur_deep_filed/download/fieldRNN_data/eval"

train_loader = Dataloader(datafolder=train_localdir, batchsize=batch_size)
test_loader = Dataloader(datafolder=test_localdir, batchsize=batch_size)
n_classes = train_loader.nclasses

#num_train_samples = 
#num_test_samples = 

channel_sizes = [args.nhid] * args.levels
kernel_size = args.ksize
model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=args.dropout)

num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('Number of params: ', str(num_param))


if args.cuda:
    model.cuda()

lr = args.lr
optimizer = getattr(optim, args.optim)(model.parameters(), lr=lr)


def train():
    train_loss = 0
    model.train()
    batch_idx = -1
    
    while train_loader.epoch < epochs :    
        batch_idx += 1 
        data, target, _ = train_loader.next_batch()
        data = data[:,:-2,:]
        target = target[:,:-2,:]
        target = np.argmax(target, axis=-1) 
        data, target = torch.from_numpy(data), torch.from_numpy(target)
        data = data.permute(0,2,1)
        data = data.type(torch.FloatTensor)        
        data, target = Variable(data.cuda()), Variable(target.cuda())
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        train_loss += loss
        if batch_idx > 0 and batch_idx % args.log_interval == 0:
            print('Train Epoch: {} \tLoss: {:.6f}\tSteps: {}'.format(
                train_loader.epoch, train_loss.item()/args.log_interval, batch_idx))
            train_loss = 0


def test():
    model.eval()
    test_loss = 0
    correct = 0
    batch_idx = -1
    num_test_samples = 0
    
    with torch.no_grad():
        while test_loader.epoch < 1 :    
            batch_idx += 1 
            data, target, _ = test_loader.next_batch()
            data = data[:,:-2,:]
            target = target[:,:-2,:]
            target = np.argmax(target, axis=-1) 
            data, target = torch.from_numpy(data), torch.from_numpy(target)
            data = data.permute(0,2,1)
            data = data.type(torch.FloatTensor)        
            data, target = Variable(data.cuda(), volatile=True), Variable(target.cuda())
            
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            num_test_samples += pred.shape[0]*pred.shape[2]
        test_loss /= num_test_samples
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, num_test_samples,
            100. * correct / num_test_samples))
        return test_loss


if __name__ == "__main__":
    train()
    test()
       