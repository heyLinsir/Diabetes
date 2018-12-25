import csv
import pickle
import random

import numpy as np

import datatype

def padding_feature(x, max_length):
    assert x.shape[0] <= max_length
    if x.shape[0] == max_length:
        return x
    return np.concatenate([x, [0.] * (max_length - x.shape[0])], axis=0)

def resample_split_set(data, path='data'):
    split_by_class = [[], [], []]
    for item in data:
        split_by_class[int(item[-1])].append(item)
    test_data = []
    train_data = []
    for i, item in enumerate(split_by_class):
        print('%d class has %d examples' % (i, len(item)))
        random.shuffle(split_by_class[i])
        test_data.extend(split_by_class[i][:1500])
        train_data.append(np.stack(split_by_class[i][1500:]))
    test_data = np.stack(test_data, axis=0)
    dev_data = test_data
    print('train num: %d' % (sum([x.shape[0] for x in train_data])))
    print('dev   num: %d' % (dev_data.shape[0]))
    print('test  num: %d' % (test_data.shape[0]))
    pickle.dump(train_data, open('%s/train.pkl' % (path), 'wb'))
    pickle.dump(dev_data, open('%s/dev.pkl' % (path), 'wb'))
    pickle.dump(test_data, open('%s/test.pkl' % (path), 'wb'))

def split_set(data, path='data'):
    random.shuffle(data)
    test_data = data[:4500]
    train_data = data[4500:]
    dev_data = test_data
    print('train num: %d' % (train_data.shape[0]))
    print('dev   num: %d' % (dev_data.shape[0]))
    print('test  num: %d' % (test_data.shape[0]))
    pickle.dump(train_data, open('%s/train.pkl' % (path), 'wb'))
    pickle.dump(dev_data, open('%s/dev.pkl' % (path), 'wb'))
    pickle.dump(test_data, open('%s/test.pkl' % (path), 'wb'))

def split_set_by_id(data, path='data'):
    '''
    5% for dev set
    5% for test set
    90% for train set
    '''
    id_allocate = {'train': [], 'dev': [], 'test': []}
    random.shuffle(data)
    train_data = []
    dev_data = []
    test_data = []

    for item in data:
        pid = int(item[-1])
        if pid in id_allocate['train']:
            train_data.append(item[:-1])
        elif pid in id_allocate['dev']:
            dev_data.append(item[:-1])
        elif pid in id_allocate['test']:
            test_data.append(item[:-1])
        else:
            choice = random.random()
            if choice < 0.9:
                train_data.append(item[:-1])
                id_allocate['train'].append(pid)
            elif choice < 0.95:
                dev_data.append(item[:-1])
                id_allocate['dev'].append(pid)
            else:
                test_data.append(item[:-1])
                id_allocate['test'].append(pid)

    train_data = np.stack(train_data, axis=0)
    dev_data = np.stack(dev_data, axis=0)
    test_data = np.stack(test_data, axis=0)

    print('train num: %d' % (train_data.shape[0]))
    print('dev   num: %d' % (dev_data.shape[0]))
    print('test  num: %d' % (test_data.shape[0]))
    pickle.dump(train_data, open('%s/train.pkl' % (path), 'wb'))
    pickle.dump(dev_data, open('%s/dev.pkl' % (path), 'wb'))
    pickle.dump(test_data, open('%s/test.pkl' % (path), 'wb'))

def normalize(data):
    max_value = data.max(axis=0) + 1e-12
    max_value[-2:] = 1
    return data / max_value

def load_data(filename='data/diabetic_data.csv', use_resample=False):
    print('begin load data from %s...' % (filename))
    used_datatypes = [datatype.features_for_medications()] # don't change(auto generated)
    id_collector = datatype.patient_nbr()
    all_datatypes = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'features_for_medications'] # don't change(auto generated)
    unused_datatypes = [] # the data not used(add name of un-used data)
    data_features = [] # don't change(auto generated)
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0: # create feature collectors
                all_datatypes = row
                print('all feature types: %s' % (str(all_datatypes)))
                for key in row:
                    if key in used_datatypes[0].key_list or key in unused_datatypes:
                        continue
                    used_datatypes.append(eval('datatype.%s' % (key))())
                if 'features_for_medications' in unused_datatypes:
                    used_datatypes = used_datatypes[1:]
            else: # load data
                item = {key: value for (key, value) in zip(all_datatypes, row)}
                data_features.append([collector(item) for collector in used_datatypes] + [id_collector(item)])
    feature_lengths = [[len(feature) for feature in item] for item in data_features]
    feature_lengths = np.max(feature_lengths, axis=0)
    data_features = [[padding_feature(feature, length) for length, feature in zip(feature_lengths, item)] for item in data_features]
    data_features = np.stack([np.concatenate(item, axis=0) for item in data_features], axis=0) # data_num * feature_size
    data_features = normalize(data_features)
    pickle.dump(data_features, open('%s.pkl' % filename, 'wb'))
    print('features embedding created!')
    print('data num    : %d' % (data_features.shape[0]))
    print('feature size: %d' % (data_features.shape[1]))
    print('use %d types of feature:' % len(used_datatypes))
    for collector, length in zip(used_datatypes, feature_lengths):
        print('\t%s feature size: \t%d' % (collector.name, length))
    print('feature saved to %s.pkl' % filename)
    split_set_by_id(data_features)
   # if use_resample:
   #     resample_split_set(data_features)
   # else:
   #     split_set(data_features)

if __name__ == '__main__':
    load_data(use_resample=False)
