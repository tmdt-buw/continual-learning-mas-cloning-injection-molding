import math
import random
import argparse
import copy
from datetime import datetime
from collections import OrderedDict

import xarray as xr

import numpy as np
import pandas as pd
import os
import time
import torch
import torch.nn as nn

from pandas_data_set import PandasDataset
from model import MLP
from routines import *

######################################
#     settings & hyperparameters     #
######################################
'''
 part_filter: 6x2_Lego_hoch 8x1_Lego_hoch 6x1_Lego 4x2_Lego 4x1_Lego 4x1_Lego_hoch 4x1_Lego_flach 3x1_Lego 3x1_Lego_hoch 8x2_Lego_flach 8x1_Lego_flach 3x2_Lego 8x2_Lego 8x1_Lego 2x2_Lego 3x2_Lego_flach 2x2_Lego_flach 6x2_Lego 6x1_Lego_flach 3x2_Lego_hoch 3x1_Lego_flach 4x2_Lego_hoch 2x1_Lego 8x2_Lego_hoch 6x2_Lego_flach 2x2_Lego_hoch 4x2_Lego_flach 6x1_Lego_hoch 2x1_Lego_flach 2x1_Lego_hoch
'''

# parse arguments
parser = argparse.ArgumentParser(
    description='Experiment script.',
    epilog="Trains an MLP model incrementally using different values for gamma and lambda.")

# settings
parser.add_argument('-exp_name', type=str, default='exp',
    help='experiment name/title that determines the output files (i.e. results, models) of the script')
parser.add_argument('-data_set', type=str, default='injection-molding',
    help="specifies which data set to use ('injection-molding' or 'glass-forming')")
parser.add_argument('-save_models', type=str, default='false',
    help='store models as part of the results')
parser.add_argument('-offload_aux_models', type=str, default='false',
    help='offload auxiliary models (i.e. base model) to file instead of storing it internally (requires more active memory)')
parser.add_argument('-use_cuda', type=str, default='false',
    help='specifies whether to use CUDA or run on CPU')
parser.add_argument('-sequence', dest='sequence_list', type=str, nargs='*', default=None,
    help='sequence(s) of tasks the experiment is performed on (may be of different lengths, ommit for random sequences)')
parser.add_argument('-n_base', type=int, default=1,
    help='number of base tasks')
parser.add_argument('-n_inc', type=int, default=-1,
    help='number of increments')
parser.add_argument('-seed', type=int, default=None,
    help='specifies a seed (for both NumPy and PyTorch)')
parser.add_argument('-n_shuffles', type=int, default=1,
    help='number of random training, validation and test set shuffles the experiment is performed on for each sequence')
parser.add_argument('-n_sequences', type=int, default=1,
    help='number of sequences the experiment is performed on (only relevant if no explicit sequences are specified)')
parser.add_argument('-part_filter', type=str, nargs="*", default=None,
    help='specifies the parts/tasks to be used from the data set (only relevant if no explicit sequences are specified)')
parser.add_argument('-use_scaling', type=str, default='false',
    help='specifies whether to use standard scaling for each part or not')
parser.add_argument('-skip_n_shuffles', type=int, default=0,
    help='skip first n shuffles')

# hyperparams
parser.add_argument('-hidden_dims', nargs='+', type=int, required=True,
    help='hidden dimensions of the MLP model (corresponds to the number of neurons in the respective layer)')
parser.add_argument('-lambda', dest='lambda_list', nargs='+', type=float, default=[1.0],
    help='lambda value(s) to perform experiment with')
parser.add_argument('-gamma', dest='gamma_list', nargs='+', type=float, default=[1.0],
    help='gamma value(s) to perform experiment with')
parser.add_argument('-batch_size', type=int, required=True,
    help='batch size to be used during training')
parser.add_argument('-lr', type=float, required=True,
    help='learning rate to be used for optimizing the model')
parser.add_argument('-test_proportion', type=float, default=0.15,
    help='proportion of the test set relative to the overall data set')
parser.add_argument('-val_proportion', type=float, default=0.1,
    help='proportion of the validation set relative to the overall data set')
parser.add_argument('-subset_fraction', type=float, default=1.0,
    help='subset fraction to be applied to the training and validation sets (test set remains unaffected!)')
parser.add_argument('-init_outputs', type=str, default='random',
    help="specifies how to initialize output headers ('random' randomly initialize, 'cloning' to clone the most similar output head or 'cloning-random' to clone a random output head)")
parser.add_argument('-target_eps', type=int, default=0.1,
    help='margin of the target loss within the binary search operates')
parser.add_argument('-target_iterations', type=int, default=1,
    help='number of iterations for data set size binary search (1 is equivalent to not using binary search and only using the full data set)')

# parse arguments and hyperparams
args = parser.parse_args()

# set device (CUDA gpu or CPU)
device = "cuda" if torch.cuda.is_available() and args.use_cuda.lower() == "true" else "cpu"
print(f'Device: {device}')

# convert arguments to boolean
args.offload_aux_models = True if args.offload_aux_models.lower() == "true" else False
args.save_models = True if args.save_models.lower() == "true" else False
args.use_scaling = True if args.use_scaling.lower() == "true" else False

# path where model is stored
model_path = '{}.pt'.format(args.exp_name)

# path where results are stored
experiment_name = datetime.now().strftime("%Y%m%d-%H%M%S")
results_directory = os.path.join("results/", experiment_name)
if not os.path.exists(results_directory):
    os.mkdir(results_directory)
results_path = os.path.join(results_directory, '{}.experiment'.format(args.exp_name))

with open(os.path.join(results_directory, 'experiment.txt'), 'w') as txt_file:
    txt_file.write(f'Experiment: MAS, args: {args}')

if(isinstance(args.lambda_list, float)):
    args.lambda_list = [args.lambda_list]
if(isinstance(args.gamma_list, float)):
    args.gamma_list = [args.gamma_list]

# set seeds for determinism and reproducibility (if specified)
if(args.seed is not None):
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# prepare data set
assert args.data_set in ['injection-molding', 'glass-forming'], "Data set {} not recognized".format(args.data_set)

if(args.data_set == 'injection-molding'):
    # path to data set
    csv_path = "data/injection_molding_lego_data_full.csv"
    csv_sep = ';'

    # input and output attributes (in specified order)
    input_attrs  = ['Kuehlzeit', 'Nachdruck', 'Schmelzetemperatur', 'Wandtemperatur', 'Volumenstrom']
    output_attrs = ['Max_Deformation']
    task_id_attr = 'Legobaustein'
elif(args.data_set == 'glass-forming'):
    # path to data set
    csv_path = "data/data_2018_2019.csv"
    csv_sep = ','

    # input and output attributes (in specified order)
    input_attrs  = ['raytek_1', 'raytek_2', 'raytek_3_inner', 'raytek_3_outer', 'raytek_4', 'raytek_6_inner', 'raytek_6_outer', 'raytek_7', 'ch15_chamber_temp', 'ch17_chamber_temp']
    output_attrs = ['geometry_q1']
    task_id_attr = 'task_id'

# check if 'init_outputs' has a valid value
assert args.init_outputs in ['random', 'cloning', 'cloning-random'], "Value {} of argument 'init_outputs is not valid".format(args.init_outputs)

##########################################
#     prepare sequences and data set     #
##########################################

# read data from CSV file
df = pd.read_csv(csv_path, sep=csv_sep)

# if no parts/tasks are specified use all parts/tasks of the data set
if(args.part_filter is None):
    args.part_filter = list(df[task_id_attr].unique())

# if no explicit sequences are specified create random sequences
if(args.sequence_list is None):

    args.sequence_list = []

    # create random sequences
    for i in range(args.n_sequences):
        sequence = random.sample(args.part_filter, len(args.part_filter))
        args.sequence_list.append(sequence)
else:
    # parse sequence(s) from the arguments
    args.sequence_list = [list(map(str, sequence.split(','))) for sequence in args.sequence_list]

# filter parts/tasks (if unspecified all parts/tasks are used)
df = df.loc[df[task_id_attr].isin(args.part_filter)]

# check that the number of base tasks and incremental tasks does not exceed the number of tasks in the sequences
assert all([args.n_base + args.n_inc <= len(sequence) if args.n_inc > 0 else args.n_base <= len(sequence) for sequence in args.sequence_list]), "Number of base tasks and increments exceeds sequence of tasks"

#######################
#     experiments     #
#######################

torch.save([], results_path)

# for each sequence
for sequence_id in range(len(args.sequence_list)):

    for shuffle_skip_id in range(args.skip_n_shuffles):
        for name, data in df.groupby(task_id_attr):
            # create dummy shuffle (to be discarded)
            indices = list(data.index)
            random.shuffle(indices)

    for shuffle_id in range(args.n_shuffles):
        start_time = time.time()

        # load sequence
        sequence = copy.deepcopy(args.sequence_list[sequence_id])

        # if no number of increments is specified (default -1) use all remaining parts/task as increments
        n_inc = len(sequence)-args.n_base if args.n_inc == -1 else args.n_inc

        print("Sequence {}: {}, shuffle {}\n".format(sequence_id+1, sequence, shuffle_id+1))

        # dictionary that translates the name of a part/task to its index in the sequence (needed to translate the (optional) task identifiers and to select the correct output heads) 
        task_id_dict = {name: i for i, name in enumerate(sequence)}

        # create training, test and validation sets
        parts = OrderedDict()

        for name, data in df.groupby(task_id_attr):

            # create actual shuffle
            indices = list(data.index)
            random.shuffle(indices)

            # split indices into training, test and validation sets (training set size is deduced from the proportions of the validation and test sets of the full data set)
            train_ids, test_ids, val_ids = map(list, np.split(indices, ((1-np.cumsum([args.val_proportion, args.test_proportion])[::-1]) * len(data)).astype(int)))

            if(args.use_scaling):
                # compute mean and standard deviation of training set
                train_data = data[data.index.isin(train_ids)][input_attrs + output_attrs]
                mean, std = train_data.mean(), train_data.std()

                # replace zeros with ones to avoid divide-by-zero errors (does not make a difference if std is zero)
                std = std.replace(0.0, 1.0)

                # standard scale all sets
                data = (data[input_attrs + output_attrs] - mean) / std

                # sanity checks
                for index, row in data.iterrows():
                    for col in input_attrs + output_attrs:
                        assert math.isclose(df.at[index, col], (row[col] * std[col]) + mean[col], rel_tol=1e-09, abs_tol=0.0), "Column %r at index %r for group %r does not match original value when rescaled: %r vs %r" % (col, index, name, df.at[index, col], (row[col] * std[col]) + mean[col])

                # replace original values with scaled ones
                df.update(data)

            # store sets
            parts[name] = (train_ids, test_ids, val_ids)

        # create PyTorch data set from pandas data frame
        data = PandasDataset(df, inputs=input_attrs, outputs=output_attrs, task_id=task_id_attr, use_pd_indices=True)

        # create base model
        model = MLP(input_dim=len(input_attrs), hidden_dims=args.hidden_dims, output_dim=len(output_attrs), n_heads=args.n_base + n_inc)
        model.to(device)

        criterion = torch.nn.MSELoss()

        #########################
        #     base training     #
        #########################

        session = []
        session_train_ids = []
        session_test_ids  = []
        session_val_ids   = []

        data_set_sizes = []
        cloned_outputs = []

        for i in range(args.n_base):
            # update session
            task = sequence.pop(0)
            session.append(task)

            # update training, testing and valudation sets
            train_ids, test_ids, val_ids = parts[task]

            session_train_ids.append(train_ids)
            session_test_ids.append(test_ids)
            session_val_ids.append(val_ids)

            data_set_sizes.append(args.subset_fraction)
            cloned_outputs.append(None)

        # create training and validation loaders
        session_train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(sum(session_train_ids, [])))
        session_val_loader   = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(sum(session_val_ids, [])))

        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

        model_path_base = "base_{}".format(model_path)

        # joint training on all base tasks
        train_loss, val_loss = train_model(model, criterion, optimizer, session_train_loader, session_val_loader, task_id_dict=task_id_dict, outpath=model_path_base, device=device, store_model_internally=True)

        # update omega values
        for task_data in session_train_ids:
            task_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(task_data))
            model.update_omega(task_loader, task_id_dict=task_id_dict, gamma=1.0, device=device)

        # update theta values
        model.update_theta()

        # create test loader for each task
        session_test_loader_list = [torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(ids)) for ids in session_test_ids]

        # evaluate model
        test_loss = eval_model(model, criterion, session_test_loader_list, task_id_dict=task_id_dict, device=device)

        # wrap test loss in list (in case a single task is evaluated a single numerical value is returned)
        if(len(session_test_loader_list) == 1):
            test_loss = [test_loss]

        # store results/stats and base network (to be reused for all successive lamda/gamma variations)
        if(args.save_models):
            base_stats = copy.deepcopy([session, train_loss, val_loss, test_loss, model.state_dict()])
        else:
            base_stats = copy.deepcopy([session, train_loss, val_loss, test_loss])

            # save reference model internally or externally
            if(args.offload_aux_models):
                model.store(model_path_base)
            else:
                model_base_state_dict = copy.deepcopy(model.state_dict())

        ######################
        #     increments     #
        ######################

        for lamb in args.lambda_list:

            for g, gamma in enumerate(args.gamma_list):

                # initialize statistics (reuse results/stats for the training of the base network)
                stats = [base_stats]

                # restore session, sequence, data set splits and model to the reference point (i.e. after training of the base network)
                session  = copy.deepcopy(args.sequence_list[sequence_id][:args.n_base])
                sequence = copy.deepcopy(args.sequence_list[sequence_id][args.n_base:]) # remaining parts/tasks (rest of the sequence)
                session_train_ids = session_train_ids[:args.n_base]
                session_test_ids  = session_test_ids[:args.n_base]
                session_val_ids   = session_val_ids[:args.n_base]

                model = MLP(input_dim=len(input_attrs), hidden_dims=args.hidden_dims, output_dim=len(output_attrs), n_heads=args.n_base + n_inc)
                model.to(device)

                # restore base model
                if(args.save_models):
                    # load model stored in statistics of the base training
                    model.load_state_dict(base_stats[-1])
                else:
                    if(args.offload_aux_models):
                        # load model from file
                        model.load(model_path_base)
                    else:
                        # load model from internal copy
                        model.load_state_dict(model_base_state_dict)

                for inc_id in range(n_inc):

                    # update session
                    new_task = sequence.pop(0)
                    session.append(new_task)

                    # update training, testing and valudation sets
                    train_ids, test_ids, val_ids = parts[new_task]

                    # scale training and validation sets to the subset fraction (if specified, defaults to 1.0)
                    train_ids = train_ids[:max(int(len(train_ids) * args.subset_fraction), 1)]
                    val_ids   =   val_ids[:max(int(  len(val_ids) * args.subset_fraction), 1)]

                    session_train_ids.append(train_ids)
                    session_test_ids.append(test_ids)
                    session_val_ids.append(val_ids)

                    data_size = 1.0
                    
                    # data set size pivots for binary search
                    upper_pivot = 1.0
                    lower_pivot = 0.0

                    reference_loss = np.Inf
                    data_size_tmp = None
                    stats_tmp = None

                    # save starting model for this increment
                    if(not args.save_models):                        
                        # save reference model internally or externally
                        if(args.offload_aux_models):
                            model_path_inc_ref = "inc_ref_{}".format(model_path)

                            model.store(model_path_inc_ref)
                        else:
                            model_inc_ref_state_dict = copy.deepcopy(model.state_dict())

                    for iteration_id in range(args.target_iterations):

                        print("gamma {}, lambda {}, inc {}, iteration {}, data size {}".format(gamma, lamb, inc_id+1, iteration_id+1, data_size))

                        # load starting model for this increment
                        if(args.save_models):
                            # load model stored in statistics after the last increment
                            model.load_state_dict(stats[-1][-1])
                        else:
                            if(args.offload_aux_models):
                                # load model from file
                                model.load(model_path_inc_ref)
                            else:
                                # load model from internal copy
                                model.load_state_dict(model_inc_ref_state_dict)

                        # create subsets of the training and validation sets
                        iter_train_ids = train_ids[:max(int(data_size*len(train_ids)), 1)]
                        iter_val_ids   = val_ids[:max(int(data_size*len(val_ids)), 1)]

                        # create training and validation loaders
                        session_train_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(iter_train_ids))
                        session_val_loader   = torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(iter_val_ids))

                        # create test loader for each task
                        session_test_loader_list = [torch.utils.data.DataLoader(data, batch_size=args.batch_size, sampler=torch.utils.data.SubsetRandomSampler(ids)) for ids in session_test_ids]

                        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

                        model_path_inc = "inc_{}".format(model_path)

                        if(args.init_outputs == 'cloning'):
                            # put model into evaluation mode
                            model.eval()

                            errors = np.zeros(len(session))

                            for t, x, y in session_train_loader:
                                x, y = x.to(device), y.to(device)

                                # compute errors for all ACTIVE (!) output heads
                                outputs = model(x, head=list(range(len(session))) )

                                # accumulate respective task errors
                                errors += np.array([criterion(out, y).item() * x.size(0) for out in outputs])

                            cloning_index = np.argmin(errors)

                            # select current output head
                            new_task_output = model.layers['out{}'.format(session.index(new_task) + 1)]
                            # select output head with least error
                            cloning_output = model.layers['out{}'.format(cloning_index + 1)]

                            # clone output weights
                            with torch.no_grad():
                                new_task_output.weight.copy_(cloning_output.weight)
                                new_task_output.bias.copy_(cloning_output.bias)

                            # put model back into training mode
                            model.train()
                        elif(args.init_outputs == 'cloning-random'):
                            # select random output head index
                            cloning_index = random.randint(0, len(session)-1)

                            # select current output head
                            new_task_output = model.layers['out{}'.format(session.index(new_task) + 1)]
                            # select output head
                            cloning_output = model.layers['out{}'.format(cloning_index + 1)]

                            # clone output weights
                            with torch.no_grad():
                                new_task_output.weight.copy_(cloning_output.weight)
                                new_task_output.bias.copy_(cloning_output.bias)
                        else:
                            cloning_index = None

                        # joint training on all base tasks
                        train_loss, val_loss = train_model(model, criterion, optimizer, session_train_loader, session_val_loader, use_omega=True, lamb=lamb, task_id_dict=task_id_dict, outpath=model_path_inc, device=device, separate_omega_loss=True, store_model_internally=True)

                        # update omega and theta values
                        model.update_omega(session_train_loader, task_id_dict=task_id_dict, gamma=gamma, device=device)
                        model.update_theta()

                        # evaluate model
                        test_loss = eval_model(model, criterion, session_test_loader_list, task_id_dict=task_id_dict, device=device)

                        # wrap test loss in list (in case a single task is evaluated a single numerical value is returned)
                        if(len(session_test_loader_list) == 1):
                            test_loss = [test_loss]

                        # first iteration
                        if(data_size == 1.0):
                            reference_loss = test_loss[-1]

                        # if the test loss is within a specified margin of the reference loss (full data set)
                        if(test_loss[-1] <= reference_loss * (1.0 + args.target_eps)):
                            if(args.save_models):
                                stats_tmp = copy.deepcopy([session, train_loss, val_loss, test_loss, model.state_dict()])
                            else:
                                stats_tmp = copy.deepcopy([session, train_loss, val_loss, test_loss])

                                # save reference model internally or externally
                                if(args.offload_aux_models):
                                    model_path_tmp = "tmp_{}".format(model_path)
                                    model.store(model_path_tmp)
                                else:
                                    model_tmp_state_dict = copy.deepcopy(model.state_dict())

                            data_size_tmp = data_size

                            upper_pivot = data_size
                            data_size = data_size - ((data_size - lower_pivot)/2.0)
                        else:
                            lower_pivot = data_size
                            data_size = data_size + ((upper_pivot - data_size)/2.0)

                        # best possible result already achieved
                        if(data_size >= 1.0):
                            data_size = 1.0
                            break

                    # restore best model/stats from binary search
                    stats.append(copy.deepcopy(stats_tmp))
                    data_set_sizes.append(data_size_tmp * args.subset_fraction)
                    cloned_outputs.append(cloning_index)

                    if(args.save_models):
                        # load model stored in statistics
                        model.load_state_dict(stats_tmp[-1])
                    else:
                        if(args.offload_aux_models):
                            # load model from file
                            model.load(model_path_tmp)
                        else:
                            # load model from internal copy
                            model.load_state_dict(model_tmp_state_dict)

                # create and append xarray data array with experiment info
                stats_xarray = xr.DataArray(stats,
                                            dims=['increment', 'stats'],
                                            coords={'stats': ['session', 'train_loss', 'val_loss', 'test_loss', 'model'] if args.save_models else ['session', 'train_loss', 'val_loss', 'test_loss']},
                                            attrs={'lr': args.lr, 'batch_size': args.batch_size, 'n_base': args.n_base, 'n_inc': n_inc, 'hidden_dims': args.hidden_dims, 'sequence': args.sequence_list[sequence_id], 'exp_name': args.exp_name, 'date': datetime.now().strftime("%m-%d-%Y (%H:%M:%S)"),
                                                   'seed': args.seed, 'shuffle_id': args.skip_n_shuffles + shuffle_id, 'test_proportion': args.test_proportion, 'val_proportion': args.val_proportion, 'subset_fraction': args.subset_fraction, 'lambda': lamb, 'gamma': gamma if lamb != 0.0 else 'none', 'data_set_sizes': data_set_sizes, 'cloned_outputs': cloned_outputs, 'init_outputs': args.init_outputs, 'target_iterations': args.target_iterations, 'target_eps': args.target_eps})

                # update stored results/stats
                experiments = torch.load(results_path)
                experiments.append(stats_xarray)
                torch.save(experiments, results_path)

                del experiments

                print(f'Experiment duration: {round(time.time() - start_time, 1)} seconds')

                # if lambda is zero then there is no need to test multiple values of gamma (reuse first stats/results instead
                if(lamb == 0):
                    break
