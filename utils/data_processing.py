import json

import numpy as np
import random
import pandas as pd
import torch.nn.functional as F
import torch


class Data:
    def __init__(self, sources, destinations, realt, labels, year, ts_all):
        self.sources = sources
        self.destinations = destinations
        self.realt = realt
        self.labels = labels
        self.n_interactions = len(sources)
        self.year = year
        self.ts_all = ts_all
        assert len(sources) == len(destinations)


def get_truth(file_name):
    FILE = './data/{}.json'.format(file_name)
    with open(FILE) as f:
        data = json.load(f)
    return data


def get_data(dataset_name, reflect_set, use_validation=False):
    FILE = './data/{}.csv'.format(dataset_name)
    graph_df = pd.read_csv(FILE)
    ref_list = []
    for ref_s in reflect_set:
        sub_file = './data/{}.csv'.format(ref_s)
        ref_df = pd.read_csv(sub_file,index_col=0)
        ref_data = ref_df.values[:, 1:]
        print(torch.tensor(list(ref_data[:,0])))
        ref_list.append(torch.cat([torch.tensor(list(ref_data[:,0])).unsqueeze(dim=-1), torch.tensor(list(ref_data[:,1])).unsqueeze(dim=-1)],dim=1))

    train_time, val_time, test_time = [20151232, 20161232, 20171232]

    # company,fields,timestamp,label,ts #
    sources = graph_df.company.values
    destinations = graph_df.fields.values
    ts_all = graph_df.timestamp.values

    dst_list = []
    for d in destinations:  # "[xxx,xxx]" or [xxx]
        d = d.strip("[]").split(",")

        d = [int(i) for i in d]
        dst_list.append(d)
    destinations = dst_list
    labels = graph_df.label.values
    # timestamps = graph_df.ts.values
    realt = graph_df.timestamp.values   #2018122508
    year = graph_df.label.values
    print(realt)

    train_mask = realt <= train_time if use_validation else realt <= val_time
    test_mask = np.logical_and(realt <= test_time, realt > val_time)
    val_mask = np.logical_and(realt <= val_time, realt > train_time) if use_validation else test_mask
    end_mask = realt > test_time

    tr_len = len(sources[train_mask])
    vd_len = len(sources[val_mask])
    end_len = len(sources[end_mask])

    realt = graph_df.timestamp2.values  # 2018122508

    print(realt.shape)
    print(sources.shape)
    full_data = Data(sources, destinations, realt, labels, year, ts_all)
    train_data = Data(sources[train_mask], destinations[:tr_len],
                      realt[train_mask], labels[train_mask], year, ts_all)

    val_data = Data(sources[val_mask], destinations[tr_len:tr_len + vd_len],
                    realt[val_mask], labels[val_mask], year, ts_all)

    test_data = Data(sources[test_mask], destinations[tr_len + vd_len:-end_len],
                     realt[test_mask], labels[test_mask], year, ts_all)

    print("The dataset has {} interactions".format(full_data.n_interactions))
    print("The training dataset has {} interactions".format(
        train_data.n_interactions))
    print("The validation dataset has {} interactions".format(
        val_data.n_interactions))
    print("The test dataset has {} interactions".format(
        test_data.n_interactions))

    return ref_list, full_data, train_data, val_data, test_data

def get_truth_patents(data, com_indx, year_idx, num_item):
    truth_list = []
    dst_node = []
    for i, com in enumerate(com_indx):
        label = False
        next_year = year_idx[i]
        his_dst = []
        for year in np.arange(0, next_year-1, 1):
            if len(data[str(com)][str(year)]) != 0:
                his_dst.extend(data[str(com)][str(year)])
        this_dst = data[str(com)][str(next_year-1)]
        his_dst = list(set(his_dst).difference(set(this_dst)))

        while not label:
            if len(data[str(com)][str(next_year)]) != 0:
                basket = torch.tensor(data[str(com)][str(next_year)])
                one_hot_items = F.one_hot(basket, num_classes=num_item)
                one_hot_basket, _ = torch.max(one_hot_items, dim=0)
                one_hot_basket = one_hot_basket
                label = True
            else:
                next_year +=1
        truth_list.append(one_hot_basket)
        dst_node.append((his_dst,this_dst))

    assert len(truth_list) == len(com_indx)
    truth = torch.stack(truth_list)
    return dst_node, truth
