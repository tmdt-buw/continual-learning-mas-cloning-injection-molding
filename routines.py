import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from itertools import count, takewhile

def train_model(model, criterion, optimizer, train_loader: DataLoader, val_loader: DataLoader, lamb=1.0, use_omega=False, task_id_dict: dict=None, epochs: int=np.Inf, patience: int=50, outpath: str='out.pt', device=None, restore_best_model=True, separate_omega_loss=False, store_model_internally=False):
    '''Training routine.

    Arguments:
        model: PyTorch model to be trained.
        criterion: Criterion to be used for training.
        optimizer: Optimizer to be used for training.
        train_loader: Data loader for the training set.
        val_loader: Data loader for the validation set.
        lamb: Optional float describing the weighting of the MAS loss (defaults to 1.0).
        use_omega: Boolean determining whether to include the MAS loss during training (True) or not (False, default).
        task_id_dict: Optional dictionary that translates task identifiers (e.g. strings or integers) to output head indices of the model (defaults to None).
        epochs: Integer describing the number of epochs the model is trained on or NumPy infinity (np.Inf) if early stopping is to be used with an indefinite number of epochs (defaults to np.Inf).
        patience: Integer describing the number of epochs to determine early stopping (epochs since last minimum w.r.t the validation loss, defaults to 50).
        outpath: String representing the path the best model should be stored to during and after the training (defaults to 'out.pt').
        device: PyTorch device that should be used for the data points (should match the device the model is one, defaults to None).
        restore_best_model: Boolean determining whether to restore the best model (i.e. the model with the minimum w.r.t. the validation loss) after early stopping (True, default) or not (False).
        separate_omega_loss: Boolean determining whether to return the objective loss and MAS loss separately (True) or to return the total loss (False, default).
        store_model_internally: Boolean determining whether to store the best model during training to a file specified by the argument 'outpath' (False, default) or to store the model internally instead (True), which results in a larger memory footprint.

    Returns:
        Tuple containing the list of training losses and the list of validation losses (if argument 'separate_omega_loss' is set to True, then the list of training losses contains tuples of the respective objective loss and MAS loss).
        Note, that the validation loss is already computed before the first epoch in case the initial model is the best model and that the corresponding training loss is set to np.Inf.
    '''

    # number of training and validation examples
    n_train = len(train_loader.sampler)
    n_val   = len(val_loader.sampler)
    
    # initialize minimum validation loss to infinity
    min_val_loss = np.Inf
    # initialize the early stopping counter
    epochs_since_min = 0
        
    # lists to store losses of all epochs
    train_losses = []
    val_losses   = []

    # store model mode for later recovery
    mode = model.training

    # create progress bars
    # epoch_bar = tqdm(total=None if np.isinf(epochs) else epochs, unit=" epochs")
    # epoch_bar.set_postfix({"Minimum val loss" : "", "Epochs since minimum" : ""})
    # epoch_bar.set_description("Epochs")
    #
    # train_bar = tqdm(total=len(train_loader), unit=" batches")
    # train_bar.set_description("Batches")

    # training epochs
    for e in count(start=0):

        # initialize training for epoch; np.Inf for 0th epoch (validation only)
        train_loss = 0.0 if e > 0 else np.Inf
        train_loss_omega = 0.0 if e > 0 else np.Inf

        # initialize validation loss for epoch
        val_loss   = 0.0

        # skip training for first iteration (validation only)
        if(e > 0):

            # set model to training mode
            model.train()
            
            # reset training progress bar
            # train_bar.n = 0
            # train_bar.last_print_n = 0
            # train_bar.refresh()
        
            # iterate over training set
            for t, x, y in train_loader:
                x, y = x.to(device), y.to(device)

                if(task_id_dict is not None):
                    # translate task ids
                    t = tuple(task_id_dict[sample] for sample in t)

                # retrieve unique task ids in batch (also the order of the output heads during forward pass)
                tasks = sorted(set(t))

                # zero gradients
                optimizer.zero_grad()

                # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
                out = model(x, head=tasks)
                out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[0]
            
                # compute objective loss
                loss = criterion(out, y)
            
                # compute omega loss (if specified)
                omega_loss = lamb * model.compute_omega_loss() if(use_omega) else torch.tensor(0.0, requires_grad=False)

                # compute total loss
                overall_loss = loss + omega_loss

                # loss is averaged so multiply it with the number of examples in this batch
                if(separate_omega_loss):
                    train_loss += loss.item() * x.size(0)
                    train_loss_omega += omega_loss.item() * x.size(0)
                else:
                    train_loss += overall_loss.item() * x.size(0)

                # compute gradients and update parameters
                overall_loss.backward()
                optimizer.step()
                
                # update progress bar
                # train_bar.update(1)

        # put model into evaluation mode
        model.eval()
        
        # iterate over validation set
        for t, x, y in val_loader:
            x, y = x.to(device), y.to(device)

            if(task_id_dict is not None):
                # translate task ids
                t = tuple(task_id_dict[sample] for sample in t)

            # retrieve unique task ids in batch (also the order of the output heads during forward pass)
            tasks = sorted(set(t))

            # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
            out = model(x, head=tasks)
            out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[0]

            # compute objective loss
            loss = criterion(out, y)

            # loss is average so multiply it with the number of examples in this batch
            val_loss += loss.item() * x.size(0)
            
        # normalize losses using the respective total numbers of samples
        train_loss /= n_train
        train_loss_omega /= n_train
        val_loss   /= n_val
        
        # append losses to lists
        if(separate_omega_loss):
            train_losses.append( (train_loss, train_loss_omega) )
        else:
            train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # new minimum w.r.t. the validation loss
        if val_loss <= min_val_loss:
            # update minimum validation loss
            min_val_loss = val_loss
            epochs_since_min = 0

            if(store_model_internally):
                model_state = model.state_dict()
            else:
                model.store(outpath)
        else:
            epochs_since_min += 1

        # don't count first iteration as training epoch (validation only)
        # if(e > 0):
            # update progress bar
            # epoch_bar.update(1)
            # epoch_bar.set_postfix({"Minimum val loss" : '{0:.6f}'.format(min_val_loss), "Epochs since minimum" : epochs_since_min})
        
        # check for early stopping or whether the number of specified epochs is reached
        if(epochs_since_min == patience or e == epochs):
            break

    # close progress bars
    # epoch_bar.close()
    # train_bar.close()

    if(restore_best_model):
        if(store_model_internally):
            model.load_state_dict(model_state)
        else:
            model.load(outpath)

    # restore model mode
    model.train(mode)
            
    return train_losses, val_losses

def eval_model(model, criterion, test_loader: DataLoader or list, task_id_dict: dict=None, device=None):
    '''Evaluation routine.

    Arguments:
        model: PyTorch model to be evaluated.
        criterion: Criterion to be used for evaluation.
        test_loader: Data loader for the test set or list of data loaders for multiple test sets (e.g. for different tasks).
        task_id_dict: Optional dictionary that translates task identifiers (e.g. strings or integers) to output head indices of the model (defaults to None).
        device: PyTorch device that should be used for the data points (should match the device the model is one, defaults to None).
        
    Returns:
        List containing test losses (in case a single test set is specified) or list containing the list of test losses for all specified test sets (if multiple test sets are specified).
    '''

    # store model mode for later recovery
    mode = model.training

    # put model into evaluation mode
    model.eval()

    # wrap data loader in a list in case a single data loader is specified
    if(isinstance(test_loader, DataLoader)):
        test_loader = [test_loader]

    test_losses = []

    # iterate over individual test sets
    for loader in test_loader:

        # number of test examples
        n_test = len(loader.sampler)
    
        # initialize test loss
        test_loss = 0.0

        # iterate over test set
        for t, x, y in loader:
            x, y = x.to(device), y.to(device)
            
            if(task_id_dict is not None):
                # translate task ids
                t = tuple(task_id_dict[sample] for sample in t)

            # retrieve unique task ids in batch (also the order of the output heads during forward pass)
            tasks = sorted(set(t))

            # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
            out = model(x, head=tasks)
            out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[0]

            # compute objective loss
            loss = criterion(out, y)

            # loss is average so multiply it with the number of examples in this batch
            test_loss += loss.item() * x.size(0)

        # normalize loss using the total number of samples
        test_loss = test_loss/n_test
        test_losses.append(test_loss)

    # restore model mode
    model.train(mode)

    # in case of multiple data loaders (tasks), return array of test loss arrays, else unpack single element
    return test_losses if len(test_losses) > 1 else test_losses[0]


def train_model_from_scratch(model, criterion, optimizer, train_loader: DataLoader, val_loader: DataLoader, task_id_dict: dict = None, epochs: int = np.Inf, patience: int = 50,
                outpath: str = 'out.pt', device=None, restore_best_model=True, separate_omega_loss=False,
                store_model_internally=False):

    # number of training and validation examples
    n_train = len(train_loader.sampler)
    n_val = len(val_loader.sampler)

    # initialize minimum validation loss to infinity
    min_val_loss = np.Inf
    # initialize the early stopping counter
    epochs_since_min = 0

    # lists to store losses of all epochs
    train_losses = []
    val_losses = []

    # store model mode for later recovery
    mode = model.training

    # create progress bars
    # epoch_bar = tqdm(total=None if np.isinf(epochs) else epochs, unit=" epochs")
    # epoch_bar.set_postfix({"Minimum val loss" : "", "Epochs since minimum" : ""})
    # epoch_bar.set_description("Epochs")
    #
    # train_bar = tqdm(total=len(train_loader), unit=" batches")
    # train_bar.set_description("Batches")

    # training epochs
    for e in count(start=0):

        # initialize training for epoch; np.Inf for 0th epoch (validation only)
        train_loss = 0.0 if e > 0 else np.Inf

        # initialize validation loss for epoch
        val_loss = 0.0

        # skip training for first iteration (validation only)
        if (e > 0):

            # set model to training mode
            model.train()

            # reset training progress bar
            # train_bar.n = 0
            # train_bar.last_print_n = 0
            # train_bar.refresh()

            # iterate over training set
            for t, x, y in train_loader:
                x, y = x.to(device), y.to(device)

                if (task_id_dict is not None):
                    # translate task ids
                    t = tuple(task_id_dict[sample] for sample in t)

                # retrieve unique task ids in batch (also the order of the output heads during forward pass)
                tasks = sorted(set(t))

                # zero gradients
                optimizer.zero_grad()

                # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
                out = model(x, head=tasks)
                out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else \
                out[0]

                # compute objective loss
                loss = criterion(out, y)

                # compute total loss
                overall_loss = loss

                # loss is averaged so multiply it with the number of examples in this batch
                if (separate_omega_loss):
                    train_loss += loss.item() * x.size(0)
                else:
                    train_loss += overall_loss.item() * x.size(0)

                # compute gradients and update parameters
                overall_loss.backward()
                optimizer.step()

                # update progress bar
                # train_bar.update(1)

        # put model into evaluation mode
        model.eval()

        # iterate over validation set
        for t, x, y in val_loader:
            x, y = x.to(device), y.to(device)

            if (task_id_dict is not None):
                # translate task ids
                t = tuple(task_id_dict[sample] for sample in t)

            # retrieve unique task ids in batch (also the order of the output heads during forward pass)
            tasks = sorted(set(t))

            # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
            out = model(x, head=tasks)
            out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[
                0]

            # compute objective loss
            loss = criterion(out, y)

            # loss is average so multiply it with the number of examples in this batch
            val_loss += loss.item() * x.size(0)

        # normalize losses using the respective total numbers of samples
        train_loss /= n_train
        val_loss /= n_val

        # append losses to lists
        if (separate_omega_loss):
            train_losses.append((train_loss))
        else:
            train_losses.append(train_loss)
        val_losses.append(val_loss)

        # new minimum w.r.t. the validation loss
        if val_loss <= min_val_loss:
            # update minimum validation loss
            min_val_loss = val_loss
            epochs_since_min = 0

            if (store_model_internally):
                model_state = model.state_dict()
            else:
                model.store(outpath)
        else:
            epochs_since_min += 1

        # don't count first iteration as training epoch (validation only)
        # if(e > 0):
        # update progress bar
        # epoch_bar.update(1)
        # epoch_bar.set_postfix({"Minimum val loss" : '{0:.6f}'.format(min_val_loss), "Epochs since minimum" : epochs_since_min})

        # check for early stopping or whether the number of specified epochs is reached
        if (epochs_since_min == patience or e == epochs):
            break

    # close progress bars
    # epoch_bar.close()
    # train_bar.close()

    if (restore_best_model):
        if (store_model_internally):
            model.load_state_dict(model_state)
        else:
            model.load(outpath)

    # restore model mode
    model.train(mode)

    return train_losses, val_losses