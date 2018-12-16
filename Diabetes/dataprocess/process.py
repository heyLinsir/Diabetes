import csv
import pickle

import datatype

def padding_feature(x, max_length):
    assert x.shape[0] > max_length
    if x.shape[0] == max_length:
        return x
    return np.concatenate([x, [0.] * (max_length - x.shape[0])], axis=0)

def load_data(filename='data/diabetic_data.csv'):
    used_datatypes = [datatype.features_for_medications()] # don't change(auto generated)
    all_datatypes = [] # don't change(auto generated)
    unused_datatypes = [] # the data not used(add name of un-used data)
    data_features = [] # don't change(auto generated)
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0: # create feature collectors
                all_datatypes = row
                for key in row:
                    if key in datatypes[0].key_list() or key in exclude_datatypes:
                        continue
                    datatypes.append(eval('datatype.%s' % (key))())
                if 'features_for_medications' in exclude_datatypes:
                    datatypes = datatypes[1:]
            else: # load data
                item = {key: value for (key, value) in zip(all_datatypes, row)}
                data_features.append([collector(item) for collector in used_datatypes])
    feature_lengths = [[len(feature) for feature in item] for item in data_features]
    feature_lengths = np.max(feature_lengths, axis=0)
    data_features = [[padding_feature(feature, length) for length, feature in zip(feature_lengths, item)] for item in data_features]
    data_features = np.stack([np.concatenate(item, axis=0) for item in data_features], axis=0) # data_num * feature_size
    pickle.dump(data_features, open('%s.pkl' % filename))
    print('features embedding created!')
    print('data num    : %d' % (data_features.shape[0]))
    print('feature size: %d' % (data_features.shape[1]))
    print('use %d types of feature:' % len(used_datatypes))
    for collector, length in zip(used_datatypes, feature_lengths):
        print('\t%s feature size: \t%d' % (collector.name, length))
    print('feature saved to %s.pkl' % filename)
