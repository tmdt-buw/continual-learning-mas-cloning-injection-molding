import torch
import torch.nn as nn
from collections.abc import Iterable

class MLP(nn.Module):
    """Multi-task multilayer perceptron model.

    Multilayer perceptron model with support for multiple output heads (tasks).
    Arbitrary number of hidden layers with individual numbers of neurons for each layer. 
    The output dimension is the same across all output heads.
    Supports forward pass of mixed batches from different tasks (using different output heads).

    Attributes:
        layers: Dictionary containing all network layers.
        layer_dims: List of tuples of input and output dimensions of hidden and output layers.
        n_hidden: Interger describing number of hidden layers.
        n_heads: Integer describing the number of current output heads (tasks).
        active_head: Integer describing the index of the active (default) output head.
        activation: PyTorch activation function.
    """

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, n_heads: int=1):
        '''Generates and initializes the network.

        Args:
            input_dim: Integer value describing the number of input features (dimensionality of the input).
            hidden_dims: List of integer values describing the number of neurons for each corresponding hidden layer.
            output_dim: Integer value describing the number of output features (dimensionality of the output).
            n_heads: Integer value describing the number of output heads to be created.
        '''

        super(MLP, self).__init__()

        self.n_hidden = len(hidden_dims)
        self.n_heads = n_heads

        # default output head
        self.active_head = 0

        # create pairs of input/output dimensions for each layer
        self.layer_dims = list(zip( [input_dim] + hidden_dims,    # input dimensions
                                    hidden_dims + [output_dim] )) # output dimension

        self.layers = nn.ModuleDict()
        
        # create hidden layers
        for i, dims in enumerate(self.layer_dims[:-1], 1):
            self.layers.add_module('h{}'.format(i), nn.Linear(*dims))
        
        # create output heads
        for i, dims in enumerate([self.layer_dims[-1]]*n_heads, 1):
            self.layers.add_module('out{}'.format(i), nn.Linear(*dims))

        # initialize omega and theta buffers for the network parameters
        self.init_omega_and_theta()

        # activation function
        self.activation = nn.ReLU()

    def init_omega_and_theta(self):
        '''Initializes omega and theta buffers for the corresponding network parameters.'''

        # known omega and theta values (in case of previously initialized buffers)
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        # initialize weight importance omega and reference weights theta associated with omega
        for name, param in self.named_parameters():
            # omega values are initialized with zeros
            if('omega_{}'.format(name.replace('.', '-')) not in omega_dict):
                self.register_buffer( 'omega_{}'.format(name.replace('.', '-')), torch.zeros_like(param, requires_grad=False) )
            # theta values are initialized using the current weight values
            if('theta_{}'.format(name.replace('.', '-')) not in theta_dict):
                self.register_buffer( 'theta_{}'.format(name.replace('.', '-')), param.clone().detach() )

    def forward(self, x, head: int or str or Iterable=None):
        '''Forward pass through the network.

        Args:
            x: Input to the network (must match dimension specified by the attribute 'input_dim').
            head: Output head(s) used in forward pass (integer for single output head, iterable of integers for multiple output heads and string containing 'all' to use all output heads; defaults to the currently active output head set using the 'active_head' attribute).

        Returns:
            Output of the network if a single output head is used and a tuple of multiple outputs if multiple output heads are used (corresponding to the values specified for the argument 'head').
        '''

        # forward pass through hidden layers
        for i in range(self.n_hidden):
            x = self.layers[ 'h{}'.format(i+1) ](x)
            x = self.activation(x)
        
        # use active head
        if(head is None):
            x = self.layers[ 'out{}'.format(self.active_head + 1) ](x)
        # use specified head
        elif(isinstance(head, int)):
            x = self.layers[ 'out{}'.format(head+1) ](x)
        # use all heads
        elif(isinstance(head, str) and head == 'all'):
            # update head and let it be handled below
            head = list(range(self.n_heads))
        # use multiple specified heads
        elif(isinstance(head, Iterable)):
            x = tuple(self.layers[ 'out{}'.format(h+1) ](x) for h in head)
        else:
            raise TypeError("Unknown type '{}' of argument 'head'".format(type(head)))

        return x

    def add_head(self, n=1):
        '''Creates and initializes additional output head(s) for the network.

        Args:
            n: Integer describing the number of output heads to be created (defaults to 1).
        '''
        
        # dimensions for output heads (input features, output features)
        dims = self.layer_dims[-1]

        # add new output head(s)
        for i in range(0, n):
            # increase number of output heads
            self.n_heads += 1
            # add new output layer 
            self.layers.add_module('out{}'.format(self.n_heads), nn.Linear(*dims))

        # initialize omega and theta of new output heads
        self.init_omega_and_theta()

    def use_head(self, i):
        '''Sets the active output head (to be used as the default for forward passes).

        Args:
            i: Integer describing the index of the output head to be used as the active output head.
        '''

        self.active_head = i

    def update_theta(self):
        '''Updates theta buffers using the current weight values.'''

        # get theta buffers
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}

        for name, param in self.named_parameters():
            # get matching theta value
            theta = theta_dict['theta_{}'.format(name.replace('.', '-'))]

            # clone current parameter values
            theta.data = param.clone().detach()

    def update_omega(self, data_loader: torch.utils.data.DataLoader, task_id_dict=None, gamma=1.0, use_task_id=True, accumulate=True, device=None):
        '''Updates omega buffers.
        
        Args:
            data_loader: PyTorch data loader to be used for calculation of omega values (should return task identifier t, as well as inputs x and outputs y).
            task_id_dict: Optional dictionary that translates task identifiers (e.g. strings or integers) to output head indices (defaults to None).
            gamma: Float value describing the decay factor for the previous omega values (defaults to 1.0).
            use_task_id: Boolean determining whether to only consider the respective outputs of the corresponding task heads (True, default) or whether to use all outputs for all inputs (False).
            accumulate: Boolean determining whether to add the new omega values to the previous ones subject to 'gamma' (True, default) or to overwrite them (False).
            device: PyTorch device that should be used for the data points (should match the device the model is one, defaults to None).
        '''

        # L2 loss (sum up individual losses, do not average them!)
        criterion = torch.nn.MSELoss(reduction='sum')

        # reset all (leftover) gradients
        for name, param in self.named_parameters():
            param.grad = None

        # save current model mode and set model to evaluation
        mode = self.training
        self.eval()

        # initialize sample counter
        n_samples = 0

        # accumulate gradients over all
        for i, (t, x, _) in enumerate(data_loader):

            if(task_id_dict is not None):
                # translate task ids
                t = tuple(task_id_dict[sample] for sample in t)

            # retrieve unique task ids in batch (also the order of the output heads during forward pass)
            tasks = sorted(set(t))

            # add number of samples in this batch
            n_samples += x.shape[0]

            x = x.to(device)

            if(use_task_id):
                out = self(x, head=tasks)

                # compose output tensor from (possibly) multi-headed output by selecting the correct output for each sample
                out = torch.stack([out[tasks.index(t_id)][i] for i, t_id in enumerate(t)], dim=0) if len(out) > 1 else out[0]
            else:
                out = self(x, head='all')

                # concatenate output of heads together so that we can measure the L2 loss of all individual outputs
                out = torch.cat(out, axis=1)

            # zero values to measure the L2 loss from
            zeros = torch.zeros(out.size()).to(device)

            loss = criterion(out, zeros)
            loss.backward()

        # get omega buffers
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}

        # average gradients over number of samples and add to omega
        for name, param in self.named_parameters():

            # get matching omega value
            omega = omega_dict['omega_{}'.format(name.replace('.', '-'))]

            # check if gradient is available (not the case for unused output heads)
            if(param.grad is not None):
                if(accumulate):
                    # decay previous omega values using gamma
                    omega.data *= gamma
                    # add new omega values
                    omega.data += torch.abs(param.grad.detach()) / n_samples
                else:
                    # overwrite omega values
                    omega.data = torch.abs(param.grad.detach()) / n_samples

            # zero gradients
            param.grad = None

        # restore model mode
        self.train(mode)

    def compute_omega_loss(self):
        '''Computes the MAS loss based on the omega and theta buffers.

        Returns:
            Float value describing the MAS loss.
        '''

        # get omega and theta buffers
        omega_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('omega')}
        theta_dict = {name: buff for name, buff in self.named_buffers() if name.startswith('theta')}
        
        # initialize MAS loss
        omega_loss = 0.0

        for name, param in self.named_parameters():

            # get matching omega and theta values
            omega = omega_dict['omega_{}'.format(name.replace('.', '-'))]
            theta = theta_dict['theta_{}'.format(name.replace('.', '-'))]

            # sum up squared differences in the parameters
            omega_loss += torch.sum( ((param-theta)**2) * omega )

        return omega_loss

    def load(self, path):
        '''Restores model state from file.
        
        Args:
            path: String representing the file path to load the model state from.
        '''

        self.load_state_dict(torch.load(path))

    def store(self, path):
        '''Stores model state to file.
        
        Args:
            path: String representing the file path to store the model state to.
        '''

        torch.save(self.state_dict(), path)