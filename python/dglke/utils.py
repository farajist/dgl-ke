# -*- coding: utf-8 -*-
#
# utils.py
#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import os
import csv
import argparse
import json
import numpy as np
import torch as th


to_device = lambda x, gpu_id: x.to(th.device('cpu')) if gpu_id == -1 else x.to(th.device('cuda:%d' % gpu_id))
none = lambda x: x
norm = lambda x, p: x.norm(p=p) ** p
get_scalar = lambda x: x.detach().item()
reshape = lambda arr, x, y: arr.view(x, y)

def get_compatible_batch_size(batch_size, neg_sample_size):
    if neg_sample_size < batch_size and batch_size % neg_sample_size != 0:
        old_batch_size = batch_size
        batch_size = int(math.ceil(batch_size / neg_sample_size) * neg_sample_size)
        print('batch size ({}) is incompatible to the negative sample size ({}). Change the batch size to {}'.format(
            old_batch_size, neg_sample_size, batch_size))
    return batch_size

def save_model(args, model, emap_file=None, rmap_file=None):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('Save model to {}'.format(args.save_path))
    model.save_emb(args.save_path, args.dataset)

    # We need to save the model configurations as well.
    conf_file = os.path.join(args.save_path, 'config.json')
    dict = {}
    config = args
    dict.update(vars(config))
    dict.update({'emp_file': emap_file,
                 'rmap_file': rmap_file})
    with open(conf_file, 'w') as outfile:
        json.dump(dict, outfile, indent=4)

def load_model_config(config_f):
    print(config_f)
    with open(config_f, "r") as f:
        config = json.loads(f.read())
        #config = json.load(f)

    print(config)
    return config

def load_raw_triplet_data(head_f=None, rel_f=None, tail_f=None, emap_f=None, rmap_f=None):
    if emap_f is not None:
        eid_map = {}
        id2e_map = {}
        with open(emap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                eid_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]

    if rmap_f is not None:
        rid_map = {}
        id2r_map = {}
        with open(rmap_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                rid_map[row[1]] = int(row[0])
                id2r_map[int(row[0])] = row[1]

    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(eid_map[id[:-1]])
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(rid_map[id[:-1]])
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(eid_map[id[:-1]])
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail, id2e_map, id2r_map

def load_triplet_data(head_f=None, rel_f=None, tail_f=None):
    if head_f is not None:
        head = []
        with open(head_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                head.append(int(id))
                id = f.readline()
        head = np.asarray(head)
    else:
        head = None

    if rel_f is not None:
        rel = []
        with open(rel_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                rel.append(int(id))
                id = f.readline()
        rel = np.asarray(rel)
    else:
        rel = None

    if tail_f is not None:
        tail = []
        with open(tail_f, 'r') as f:
            id = f.readline()
            while len(id) > 0:
                tail.append(int(id))
                id = f.readline()
        tail = np.asarray(tail)
    else:
        tail = None

    return head, rel, tail

def load_raw_emb_mapping(map_f):
    assert map_f is not None
    id2e_map = {}
    with open(map_f, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            id2e_map[int(row[0])] = row[1]

    return id2e_map


def load_raw_emb_data(file, map_f=None, e2id_map=None):
    if map_f is not None:
        e2id_map = {}
        id2e_map = {}
        with open(map_f, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                e2id_map[row[1]] = int(row[0])
                id2e_map[int(row[0])] = row[1]
    elif e2id_map is not None:
        id2e_map = [] # dummpy return value
    else:
        assert False, 'There should be an ID mapping file provided'

    ids = []
    with open(file, 'r') as f:
        line = f.readline()
        while len(line) > 0:
            ids.append(e2id_map[line[:-1]])
            line = f.readline()
        ids = np.asarray(ids)

    return ids, id2e_map, e2id_map

def load_entity_data(file=None):
    if file is None:
        return None

    entity = []
    with open(file, 'r') as f:
        id = f.readline()
        while len(id) > 0:
            entity.append(int(id))
            id = f.readline()
    entity = np.asarray(entity)
    return entity

def prepare_save_path(args):
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    folder = '{}_{}_'.format(args.model_name, args.dataset)
    n = len([x for x in os.listdir(args.save_path) if x.startswith(folder)])
    folder += str(n)
    args.save_path = os.path.join(args.save_path, folder)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)


def set_seed(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

def evaluate_best_result(model_name, dataset, save_path, threshold=3):
    file_pattern = '{}/{}_{}_*/result.txt'.format(save_path, model_name, dataset)
    files = glob.glob(file_pattern)
    best_result = None
    best_dir = None
    for file in files:
        dir = file.split('/')[-2]
        with open(file, 'r') as f:
            result = json.load(f)
        if best_result is None:
            best_result = result
            best_dir = dir
            continue
        else:
            cnt = 0
            for k in result.keys():
                if k == 'MR':
                    if result[k] <= best_result[k]:
                        cnt += 1
                else:
                    if result[k] >= best_result[k]:
                        cnt += 1
            if cnt >= threshold:
                best_result = result
                best_dir = dir
    print(f'''{model_name} training on {dataset} best result is in folder {best_dir}\n'
          best result:\n''')
    for k, v in best_result.items():
        print(f'{k}: {v}')

class CommonArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(CommonArgParser, self).__init__()

        self.add_argument('--model_name', default='TransE',
                          choices=['TransE', 'TransE_l1', 'TransE_l2', 'TransR',
                                   'RESCAL', 'DistMult', 'ComplEx', 'RotatE',
                                   'SimplE','ConvE'],
                          help='The models provided by DGL-KE.')
        self.add_argument('--data_path', type=str, default='data',
                          help='The path of the directory where DGL-KE loads knowledge graph data.')
        self.add_argument('--dataset', type=str, default='FB15k',
                          help='The name of the builtin knowledge graph. Currently, the builtin knowledge '\
                                  'graphs include FB15k, FB15k-237, wn18, wn18rr and Freebase. '\
                                  'DGL-KE automatically downloads the knowledge graph and keep it under data_path.')
        self.add_argument('--format', type=str, default='built_in',
                          help='The format of the dataset. For builtin knowledge graphs, '\
                                  'the foramt should be built_in. For users own knowledge graphs, '\
                                  'it needs to be raw_udd_{htr} or udd_{htr}.')
        self.add_argument('--data_files', type=str, default=None, nargs='+',
                          help='A list of data file names. This is used if users want to train KGE '\
                                  'on their own datasets. If the format is raw_udd_{htr}, '\
                                  'users need to provide train_file [valid_file] [test_file]. '\
                                  'If the format is udd_{htr}, users need to provide '\
                                  'entity_file relation_file train_file [valid_file] [test_file]. '\
                                  'In both cases, valid_file and test_file are optional.')
        self.add_argument('--delimiter', type=str, default='\t',
                          help='Delimiter used in data files. Note all files should use the same delimiter.')
        self.add_argument('--save_path', type=str, default='ckpts',
                          help='the path of the directory where models and logs are saved.')
        self.add_argument('--no_save_emb', action='store_true',
                          help='Disable saving the embeddings under save_path.')
        self.add_argument('--max_step', type=int, default=80000,
                          help='The maximal number of steps to train the model. '\
                                  'A step trains the model with a batch of data.')
        self.add_argument('--batch_size', type=int, default=1024,
                          help='The batch size for training.')
        self.add_argument('--batch_size_eval', type=int, default=8,
                          help='The batch size used for validation and test.')
        self.add_argument('--neg_sample_size', type=int, default=256,
                          help='The number of negative samples we use for each positive sample in the training.')
        self.add_argument('--neg_deg_sample', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the training. '\
                                  'When this option is turned on, the number of negative samples per positive edge '\
                                  'will be doubled. Half of the negative samples are generated uniformly while '\
                                  'the other half are generated proportional to vertex degree.')
        self.add_argument('--neg_deg_sample_eval', action='store_true',
                          help='Construct negative samples proportional to vertex degree in the evaluation.')
        self.add_argument('--neg_sample_size_eval', type=int, default=-1,
                          help='The number of negative samples we use to evaluate a positive sample.')
        self.add_argument('--eval_percent', type=float, default=1,
                          help='Randomly sample some percentage of edges for evaluation.')
        self.add_argument('--no_eval_filter', action='store_true',
                          help='Disable filter positive edges from randomly constructed negative edges for evaluation')
        self.add_argument('-log', '--log_interval', type=int, default=1000,
                          help='Print runtime of different components every x steps.')
        self.add_argument('--eval_interval', type=int, default=10000,
                          help='Print evaluation results on the validation dataset every x steps '\
                                  'if validation is turned on')
        self.add_argument('--test', action='store_true',
                          help='Evaluate the model on the test set after the model is trained.')
        self.add_argument('--num_proc', type=int, default=1,
                          help='The number of processes to train the model in parallel. '\
                                  'In multi-GPU training, the number of processes by default is set to match the number of GPUs. '\
                                  'If set explicitly, the number of processes needs to be divisible by the number of GPUs.')
        self.add_argument('--num_thread', type=int, default=1,
                          help='The number of CPU threads to train the model in each process. '\
                                  'This argument is used for multiprocessing training.')
        self.add_argument('--force_sync_interval', type=int, default=-1,
                          help='We force a synchronization between processes every x steps for '\
                                  'multiprocessing training. This potentially stablizes the training process '
                                  'to get a better performance. For multiprocessing training, it is set to 1000 by default.')
        self.add_argument('--hidden_dim', type=int, default=400,
                          help='The embedding size of relation and entity')
        self.add_argument('--lr', type=float, default=0.01,
                          help='The learning rate. DGL-KE uses Adagrad to optimize the model parameters.')
        self.add_argument('-g', '--gamma', type=float, default=12.0,
                          help='The margin value in the score function. It is used by TransX and RotatE.')
        self.add_argument('-de', '--double_ent', action='store_true',
                          help='Double entitiy dim for complex number or canonical polyadic. It is used by RotatE and SimplE.')
        self.add_argument('-dr', '--double_rel', action='store_true',
                          help='Double relation dim for complex number or canonical polyadic. It is used by RotatE and SimplE')
        self.add_argument('-adv', '--neg_adversarial_sampling', action='store_true',
                          help='Indicate whether to use negative adversarial sampling. '\
                                  'It will weight negative samples with higher scores more.')
        self.add_argument('-a', '--adversarial_temperature', default=1.0, type=float,
                          help='The temperature used for negative adversarial sampling.')
        self.add_argument('-rc', '--regularization_coef', type=float, default=0.000002,
                          help='The coefficient for regularization.')
        self.add_argument('-rn', '--regularization_norm', type=int, default=3,
                          help='norm used in regularization.')
        self.add_argument('-pw', '--pairwise', action='store_true',
                          help='Indicate whether to use pairwise loss function. '
                               'It compares the scores of a positive triple and a negative triple')
        self.add_argument('--loss_genre', default='Logsigmoid',
                          choices=['Hinge', 'Logistic', 'Logsigmoid', 'BCE'],
                          help='The loss function used to train KGEM.')
        self.add_argument('-m', '--margin', type=float, default=1.0,
                          help='hyper-parameter for hinge loss.')
         # args for ConvE
        self.add_argument('--tensor_height', type=int, default=10,
                          help='Tensor height for ConvE. Note hidden_dim must be divisible by it')
        self.add_argument('--dropout_ratio', type=float, nargs='+', default=0,
                          help='Dropout ratio for input, conv, linear respectively. If 0 is specified, ConvE will not use dropout for that layer')
        self.add_argument('--batch_norm', '-bn', type=bool, default=True,
                          help='Whether use batch normalization in ConvE or not')
        self.add_argument('--label_smooth', type=float, default=.0,
                          help='use label smoothing for training.')
        # args for reproducibility
        self.add_argument('--seed', type=int, default=0,
                          help='Random seed for reproducibility')
        self.add_argument('--num_node', type=int, default=1,
                          help='Number of node used for distributed training')
        # this is used for distributed training. not implemented yet
        self.add_argument('--node_rank', type=int, default=0,
                          help='The rank of node, ranged from [0, num_node - 1]')
        # self.add_argument('--eval_chunk', type=int, default=8,
        #                   help='Number of chunk to corrupt for the whole graph to pervent OOM for evaluation. The smaller the more RAM it consumed.')
        self.add_argument('--mode', type=str, default='fit',
                          choices=['fit', 'eval'],
                          help='Whether to train the model or to evaluate.')
        # TODO: lingfei - use function to substitute brute force sampling
        self.add_argument('--init_strat', type=str, default='uniform',
                          choices=['uniform', 'xavier', 'constant'],
                          help='Initial strategy for embeddings.')
        self.add_argument('--num_workers', type=int, default=8,
                          help='Number of process to fetch data for training/validation dataset.')

        # hyper-parameter for hyperbolic embeddings
        self.add_argument('--init_scale', type=float, default=0.001,
                          help='Initialization scale for entity embedding, relation embedding, curvature, attention in hyperbolic embeddings')
        self.add_argument('--optimizer', type=str, default='Adagrad',
                          choices=['Adagrad', 'Adam'],
                          help='Optimizer for kg embeddings')
        self.add_argument('--no_save_log', action='store_false', dest='save_log',
                          help='If specified, dglke will not save log and result file to save path.')
        self.add_argument('--tqdm', action='store_true', dest='tqdm',
                          help='Use tqdm to visualize training and evaluation process. Note this might drag speed of process 0 for multi-GPU training.')