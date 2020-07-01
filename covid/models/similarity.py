    

import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from collections import Counter
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from models.utils import get_device

def normalize_embed(X):
    '''
    
    Parameters
    ----------
    X: tensor of embeddings (embed count, embed dim)
    
    Returns
    -------
    tensor of embeddings (embed count, embed dim)
    
    '''
    
    norm = X.norm(p='fro', dim=1)[:, None]
    return X / norm

                

class SimDataset(Dataset):
    def __init__(self, X):
        super(SimDataset, self).__init__()
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def get_sim_summary(X, norm=False, batch_size=100, num_workers=8, path=None):
    '''
    
    Parameters
    ----------
    X: tensor of embeddings (embed count, embed dim)
    
    Returns
    -------
    tensor of embeddings (embed count, embed count)
    
    '''
    
    n_digits = 2
    
    logging.info('')
    logging.info('Document similarity')

    # Get device
    device = get_device()
    
    # Convert to tensor
    X = torch.Tensor(X)
    
    # Normalize embeddings
    if norm:
        X = normalize_embed(X)

    # Get dimensionality
    embed_count, embed_dim = X.shape
    logging.info('Embed count x embed dim: {} x {}'.format(embed_count, embed_dim))        


    dataset = SimDataset(X)

    dataloader = DataLoader(dataset, \
                        batch_size = batch_size, 
                        shuffle = False, 
                        num_workers = num_workers)

    # Create progress bar
    pbar = tqdm(total=embed_count)
    
    # Loop on batches
    X_trans = X.transpose(0,1).to(device)

    counter = Counter()
    for i, X_bat in enumerate(dataloader):

        # Move to device
        X_bat = X_bat.to(device)

        # Calculate similarity score across documents
        sim = torch.mm(X_bat, X_trans)

        mask = torch.ones_like(sim)


        # Calculate dist
        rounded = torch.round(sim * 10**n_digits) / (10**n_digits)
        rounded = rounded.view(-1,1).squeeze(1)

        counter.update(rounded.tolist())

        # Provide summary info
        if i == 0:
            logging.info('')
            logging.info('X_trans shape:\t{}'.format(X_trans.shape))
            logging.info('X_bat shape:\t{}'.format(X_bat.shape))
            logging.info('sim shape:\t{}'.format(sim.shape))

            logging.info('rounded shape:\t{}'.format(rounded.shape))
        
        
        # Update progress bar
        pbar.update()    
    pbar.close()     

    



    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df = df.rename(columns={'index':'similarity', 0:'count'})
    df.sort_values('similarity', inplace=True, ascending=True)
    logging.info('\n{}\n{}'.format('dist', df))

    if path is not None:
        scat_plt = df.plot(x ='similarity', y='count', kind = 'scatter', logy=True)
        #fig = plt[0].get_figure()
        fn = os.path.join(path, 'similarity.png')
        #fig.savefig(fn)
        fig = scat_plt.get_figure()
        fig.savefig(fn)
        #plt.savefig(fn)
    
    
    
    return df
    
    