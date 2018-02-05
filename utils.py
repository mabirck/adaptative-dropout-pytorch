import os, csv


def saveLog(test_loss, test_acc, correct, standout, args, epoch):
    path = './log/'
    #path += "_".join([args.arc, str(args.epochs), args.filter_reg, str(args.phi), 'seed', str(args.seed), 'depth', str(args.depth), args.intra_extra])
    path+= standout+'_MNIST_'+str(args.seed)
    path = path+'.csv'
    if epoch == 0 and os.path.isfile(path): os.remove(path)
    assert not(os.path.isfile(path) == True and epoch ==0), "That can't be right. This file should not be here!!!!"
    fields = ['epoch', epoch, 'test_loss', test_loss, 'test_acc', test_acc, 'correct', correct]
    with open(path, 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
