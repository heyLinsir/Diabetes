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
parser.add_argument("--log_per_updates", help="log model loss per x updates (mini-batches)", type=int, default=500)
args = parser.parse_args()        

def resample(train):
    results = []
    max_num = max([len(item) for item in train])
    for item in train:
        idx = np.random.randint(0, len(item), (max_num))
        results.append(item[idx])
    results = np.concatenate(results, axis=0).astype('float32')
    np.random.shuffle(results)
    results = torch.from_numpy(results)
    if not args.no_cuda:
        results = results.cuda()
    return results

def shuffle(train):
    results = train.astype('float32')
    np.random.shuffle(results)
    results = torch.from_numpy(results)
    if not args.no_cuda:
        results = results.cuda()
    return results

def calc_F1(pred, label):
    num_right = [1e-12] * 3
    num_pred = [1e-12] * 3
    num_label = [1e-12] * 3
    for p, l in zip(pred, label):
        num_pred[p] += 1.
        num_label[l] += 1.
        if p == l:
            num_right[p] += 1.
    for r, p, l in zip(num_right, num_pred, num_label):
        precise = r / p
        recall = r / l
        f1 = 2. * (precise * recall) / (precise + recall)
        print('precise: %f, recall: %f, f1: %f' % (precise, recall, f1))

def main(pathname='data', use_resample=False):
    print('begin loading data...')
    origin_train_data = pickle.load(open('%s/train.pkl' % (pathname), 'rb'))
    dev_data = pickle.load(open('%s/dev.pkl' % (pathname), 'rb')).astype('float32')
    test_data = pickle.load(open('%s/test.pkl' % (pathname), 'rb')).astype('float32')
    dev_size = dev_data.shape[0]
    test_size = test_data.shape[0]

    dev_data = torch.from_numpy(dev_data)
    test_data = torch.from_numpy(test_data)

    print('begin building model...')
    if args.model == 'mlp':
        model = MLP.MLP(input_size=dev_data.shape[1] - 1)
    else:
        print('model name error. Please input with --model mlp')

    if not args.no_cuda:
        model = model.cuda()
        dev_data = dev_data.cuda()
        test_data = test_data.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    ceriation = nn.CrossEntropyLoss()

    batch_size = args.batch_size
    dev_batch_num = int(dev_size / batch_size)
    test_batch_num = int(test_size / batch_size)

    print('begin training...')
    for e in range(args.epoch):
        if use_resample:
            train_data = resample(origin_train_data)
        else:
            train_data = shuffle(origin_train_data)
            
        train_batch_num = int(train_data.shape[0] / batch_size)

        avg_acc, avg_loss = 0., 0.
        for i in range(train_batch_num):
            optimizer.zero_grad()
            batch = train_data[i * batch_size:(i + 1) * batch_size, :-1]
            label = train_data[i * batch_size:(i + 1) * batch_size, -1].long()
            prob, prediction = model(batch, is_training=False)
            loss = ceriation(prob, label)
            avg_acc += torch.eq(prediction, label).cpu().data.numpy().mean()
            avg_loss += loss
            loss.backward()
            optimizer.step()
            if (i + 1) % args.log_per_updates == 0:
                print('train---epoch %d step %d, acc=%f, loss=%f' % (e, i, avg_acc / args.log_per_updates, avg_loss / args.log_per_updates))
                avg_acc, avg_loss = 0., 0.

        avg_acc, avg_loss = 0., 0.
        predictions = []
        labels = []
        for i in range(dev_batch_num):
            batch = dev_data[i * batch_size:(i + 1) * batch_size, :-1]
            label = dev_data[i * batch_size:(i + 1) * batch_size, -1].long()
            prob, prediction = model(batch, is_training=False)
            loss = ceriation(prob, label)
            predictions.append(prediction.cpu().data.numpy())
            labels.append(label.cpu().data.numpy())
            avg_acc += torch.eq(prediction, label).cpu().data.numpy().mean()
            avg_loss += loss
        calc_F1(np.concatenate(predictions, axis=0), np.concatenate(labels, axis=0))
        print('valid---epoch %d, acc=%f, loss=%f' % (e, avg_acc / dev_batch_num, avg_loss / dev_batch_num))

if __name__ == '__main__':
    main(use_resample=False)        
