import torch
import pandas as pd
from collections.abc import Iterable

class PandasDataset(torch.utils.data.Dataset):
    '''PyTorch data set using a Pandas data frame.
    
    '''
    
    def __init__(self, data, inputs: Iterable, outputs: Iterable, task_id=None, use_pd_indices=False):
        '''Initializes the data set.

        Attributes:
            indices: Original data point indices taken from the Pandas data frame.
            samples: Data points / samples extracted from the Pandas data frame.

        Args:
            data: Pandas data frame containing the data set.
            inputs: Iterable describing the data frame colums to be used as input features.
            outputs: Iterable describing the data frame columns to be used as output features.
            task_id: Optional argument describing the data frame column to be used as a respective task id for each data point (defaults to None).
            use_pd_indices: Boolean determining whether to use ascending index values for the data point in the order of their occurrence (False, default) or to use the indices specified in the data frame (True).
        '''

        super(PandasDataset, self).__init__()

        self.use_pd_indices = use_pd_indices

        # store data frame indices
        self.indices = list(data.index)

        # initialize data points / samples
        self.samples = []

        for _, row in data.iterrows():
            # extract input features
            x = torch.tensor([row[i] for i in inputs])
            # extract output features
            y = torch.tensor([row[i] for i in outputs])

            # extract task identifier (if specified)
            if(task_id is not None):
                self.samples.append( (row[task_id], x, y) )
            else:
                self.samples.append( (x,y) )

    def __len__(self):
        '''Returns the number of data points / samples in the data set.''' 
        
        return len(self.samples)

    def __getitem__(self, idx):
        '''Returns requested data point / sample given it's index.
        
        Args:
            idx: index of requested data point / sample.
        '''

        return self.samples[ self.indices.index(idx) ] if(self.use_pd_indices) else self.samples[idx]