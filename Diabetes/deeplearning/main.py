import pickle
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import MLP

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", help="batch size", type=int, default=64)
parser.add_argument("--epoch", help="max training epoch num", type=int, default=50)
parser.add_argument("--model", help="model name", type=str, default='mlp')
parser.add_argument("--no_cuda", help="don't use GPU", action='store_true')
parser.add_argument("--log_per_updates", help="log model loss per x updates (mini-batches)", type=int, default=100)
args = parser.parse_args()        

def main(filename='data/diabetic_data.csv.pkl'):
    print('begin loading data...')
    data = pickle.load(open(filename, 'rb')).astype('float32')
    np.random.shuffle(data)
    data_size = data.shape[0]
    dev_size = int(data_size * 0.05)
    test_size = int(data_size * 0.05)
    train_size = data_size - dev_size - test_size

    data = torch.from_numpy(data)
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[-test_size:]

    print('begin building model...')
    if args.model == 'mlp':
        model = MLP.MLP(input_size=data.shape[1] - 1)
    else:
        print('model name error. Please input with --model mlp')

    if not args.no_cuda:
        model = model.cuda()
        train_data = train_data.cuda()
        dev_data = dev_data.cuda()
        test_data = test_data.cuda()

    optimizer = optim.Adam(model.parameters())
    ceriation = nn.CrossEntropyLoss()

    batch_size = args.batch_size
    train_batch_num = int(train_size / batch_size)
    dev_batch_num = int(dev_size / batch_size)
    test_batch_num = int(test_size / batch_size)

    print('begin training...')
    for e in range(args.epoch):
        np.random.shuffle(train_data)

        avg_acc, avg_loss = 0., 0.
        for i in range(train_batch_num):
            optimizer.zero_grad()
            batch = train_data[i * batch_size:(i + 1) * batch_size, :-1]
            label = train_data[i * batch_size:(i + 1) * batch_size, -1].long()
            prob, prediction = model(batch)
            # if i == 500:
            #     print('[sample]')
            #     print('prediction: %s' % (str(prediction.cpu().data.numpy())))
            #     print('prob      : %s' % (str(prob.cpu().data.numpy())))
            #     print('label     : %s' % (str(label.cpu().data.numpy())))
            loss = ceriation(prob, label)
            avg_acc += torch.eq(prediction, label).cpu().data.numpy().mean()
            avg_loss += loss
            loss.backward()
            optimizer.step()
            if i % args.log_per_updates == 0:
                print('train---epoch %d step %d, acc=%f, loss=%f' % (e, i, avg_acc / args.log_per_updates, avg_loss / args.log_per_updates))
                avg_acc, avg_loss = 0., 0.

        avg_acc, avg_loss = 0., 0.
        for i in range(dev_batch_num):
            batch = dev_data[i * batch_size:(i + 1) * batch_size, :-1]
            label = dev_data[i * batch_size:(i + 1) * batch_size, -1].long()
            prob, prediction = model(batch)
            loss = ceriation(prob, label)
            avg_acc += torch.eq(prediction, label).cpu().data.numpy().mean()
            avg_loss += loss
        print('valid---epoch %d, acc=%f, loss=%f' % (e, avg_acc / (dev_batch_num * batch_size), avg_loss / (dev_batch_num * batch_size)))

if __name__ == '__main__':
    main()        
