import pandas as pd
import numpy as np
import os
from collections import Counter
import logging
from copy import deepcopy
from utils.misc import dict_to_list, list_to_dict
from collections import OrderedDict
from itertools import combinations, cycle
from pathlib import Path



import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


from constants import *

SIM_TYPE_AVG = 'avg'
SIM_TYPE_MAX = 'max'

H_TYPE_SUM = 'sum'
H_TYPE_LOOP = 'loop'


def entropy_as_dict_array(entropy, entropy_type):

    # Number list of dictionary to dictionary of lists
    entropy = list_to_dict(entropy)
    
    # Loop through determinate types
    if entropy_type == H_TYPE_LOOP:
        entropy = {evt_typ: np.array(h) for evt_typ, h in entropy.items()}
    
    # Sum entropy scores across determinants
    elif entropy_type == H_TYPE_SUM:    
        H = [np.array(h) for _, h in entropy.items()]
        H = np.sum(np.stack(H, axis=-1), axis=-1)
        entropy = {'all': H}
    else:
        raise ValueError("Invalid entropy_type:\t{}".format(entropy_type))    
    
    return entropy


def prob_as_dict_array(prob=None, n=None):
    
    assert (prob is not None) or (n is not None)
    
    if prob is None:
        prob_out = {'p':np.array([0]*n)}
    else:

        prob_out = []
        for doc in prob:
            d = OrderedDict()
            for evt_typ, evt_prob in doc.items():
                for lab, p in evt_prob.items():
                    d['P({}-{})'.format(evt_typ, lab)] = p
            prob_out.append(d)
        
        prob_out = list_to_dict(prob_out)
        prob_out = {evt_lab: np.array(p) for evt_lab, p in prob_out.items()}

    return prob_out

def docs_as_array(docs=None, n=None):
    
    assert (docs is not None) or (n is not None)
    
    # Convert tokenized documents to string
    if docs is None:
        docs = ['<<no text>>']*n
    else:
        docs = ['\n'.join([' '.join(sent) for sent in doc]) \
                                                        for doc in docs]

    return np.array(docs)



def plot_summary(diversity, diversity_mod, entropy, scores, path, n_top=None, sort_scores=True):
    
    d = OrderedDict()
    d['diversity'] = diversity
    d['diversity_mod'] = diversity_mod
    d['entropy'] = entropy
    d['score'] = scores
    df = pd.DataFrame.from_dict(d)
    if sort_scores:
        df.sort_values('score', inplace=True, ascending=False)
    df['document'] = list(range(len(df)))

    if n_top is not None:
        df = df.head(n_top)

    # Create plot
    colors = ['tab:blue',  'tab:green', 'tab:orange', 'black']
    y =      ['diversity', 'diversity_mod', 'entropy',    'score']
    plot = df.plot(x ='document', y=y, kind='line', color=colors)
    
    # Save plot
    fig = plot.get_figure()
    plt.tight_layout()
    fig.savefig(path)
    plt.close()

    return True             

def tabular_summary(diversity, diversity_mod, prob, entropy, scores, docs, path, n_sample=30, sort_scores=True):
    
    d = OrderedDict()
    d['diversity'] = diversity
    d['diversity_mod'] = diversity_mod
    d['entropy'] = entropy

    d['score'] = scores
    d['docs'] = docs
    for k, v in prob.items():
        d[k] = v
    df = pd.DataFrame.from_dict(d)
    if sort_scores:
        df.sort_values('score', inplace=True, ascending=False)

    if n_sample >= len(diversity):
        nth = 1
    else:
        nth = int(len(diversity)/n_sample)

    df = df.iloc[::nth]
    
    df.to_csv(path)

    return True  
'''
def tabular_summary(diversity, diversity_mod, entropy, scores, docs, path, n_sample=20, sort_scores=True):
    tabular_summary(diversity, diversity_mod, entropy, scores, docs, fn)
    d = OrderedDict()
    d['diversity'] = diversity
    d['diversity_mod'] = diversity_mod
    d['entropy'] = entropy
    d['score'] = scores
    df = pd.DataFrame.from_dict(d)
    
    
    if sort_scores:
        df.sort_values('score', inplace=True, ascending=False)
    
    nth = int(len(diversity)/n_sample)

    df = df.iloc[::nth]
    
    df.to_csv(path)

    return True  

'''

def plot_sim_combos(embeddings, path, decimals=1):

    indices = list(range(len(embeddings)))

    sim_scores = []
    combos = combinations(indices, 2)
    for idx1, idx2 in combos:
        sim = np.dot(embeddings[idx1], embeddings[idx2])
        
        sim_scores.append(np.round(sim, decimals))
    
    counter = Counter(sim_scores)
    df = pd.DataFrame.from_dict(counter, orient='index').reset_index()
    df = df.rename(columns={'index':'similarity', 0:COUNT})
    df.sort_values('similarity', inplace=True, ascending=True)
    p = Path(path)
    df.to_csv(p.with_suffix('.csv'))
    # Create plot
    plot = df.plot(x ='similarity', y=COUNT, kind='bar')
    
    # Save plot
    fig = plot.get_figure()
    fig.savefig(path)
    plt.close()

    return True       

def plot_entropy(H, path, n_top=None):
    '''
    Plot entropy arrays
    '''
    
    # Create data frame
    df = pd.DataFrame.from_dict(H)
    
    # Calculate average entropy across event types
    df['average'] = df.mean(axis=1)

    # Sort by average entropy
    df.sort_values('average', inplace=True, ascending=True)
    
    # Include index for plotting
    df['document'] = list(range(len(df)))

    if n_top is not None:
        df = df.tail(n_top)

    # Define colors
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    colors = colors[:len(H)] + ['black']
        
    # Create plot
    plot = df.plot(x ='document', y=list(H.keys()) + ['average'], kind='line', color=colors)

    # Save plot
    fig = plot.get_figure()
    fig.savefig(path)
    plt.close()
    
    return True    



def plot_label_dist(df, path, map_={OUTSIDE:UNKNOWN}):

    

    

    evt_types = df[EVENT_TYPE].unique()
    rounds = df[ROUND].unique()
    rounds.sort()
    
    n_evt_type = len(evt_types)
    n_round = len(rounds)


    df_label = df.groupby(LABEL).sum()
    df_label.sort_values(COUNT, inplace=True, ascending=False)
    order = df_label.index.values


    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(order,range(len(order))))

    fig, axs = plt.subplots(n_round, n_evt_type, figsize=(10, 4),  sharex='col', sharey='all')

    
    for i, evt_type in enumerate(evt_types):



        df_evt = df[df[EVENT_TYPE] == evt_type]
        
        
        
        labels = df_evt[LABEL].unique()

        
        for j, round_ in enumerate(rounds):
            df_rnd = df_evt[df_evt[ROUND] == round_]
            
            for lab in labels:
                if lab not in df_rnd[LABEL].unique():
                    d = {EVENT_TYPE: evt_type, LABEL:  lab, COUNT:0}
                    df_rnd = df_rnd.append(d, ignore_index = True) 
            

            #df_rnd = df_rnd.sort_values(LABEL)     
            df_rnd['rank'] = df_rnd[LABEL].map(sorterIndex)
            df_rnd = df_rnd.sort_values('rank')  
            df_rnd.drop('rank', 1, inplace=True)   
            
            
            df_rnd[LABEL].replace(map_, inplace=True)
            
            if n_round == 1:
                ax = axs[i]
            else:
                ax = axs[j, i]
            
            label_pos = np.arange(len(df_rnd[LABEL]))
            height = df_rnd[COUNT]
            ax.bar(label_pos, height)

            plt.sca(ax)
            plt.xticks(label_pos, df_rnd[LABEL], rotation='vertical')
            
            if j == 0:
                ax.set_title(evt_type)
            if i == 0:
                ax.set_ylabel('Round {}'.format(j+1), rotation=0, size='large', horizontalalignment='right')

    
    #fn = os.path.join(path, 'active_learning_label_dist.png')
    fig.tight_layout()
    fig.savefig(path)
    plt.close()    

def active_sample(sample_count, ids, embed, entropy, \
        entropy_type = H_TYPE_LOOP,
        sim_type = SIM_TYPE_MAX,
        alpha = 1.0,
        path = None,
        docs = None, 
        prob = None,
        n_examples = None,
        entropy_min = 0,
        div_min = 1e-20):
    '''
    Actively select samples for annotation, 
    based on sample similarity and entropy
    
    
    Parameters
    ----------
    sample_count: number of samples to select for annotation
    ids: list of unique sample identifiers, 
            e.g. [1, 123, 23421, 90...]
    embed: list of document embeddings, 
            e.g. [[0.6564, 0.0156, ...],...[0.9413, 0.1654]]
    entropy: list of dictionary with entropy scores by event type
            e.g. [{'Alcohol': 0.35, 'Drug': 2.15,...},...{'Alcohol': 0.49, 'Drug': 1.15,...}]
    sim_type: similarity type as string, ('avg'|'max')
    alpha: power for diversity score as float
    path: directory for saving summary information
    
    Returns
    -------
    sampled_ids: list of ids to annotate
    '''

    logging.info("")
    logging.info("="*72)
    logging.info("Active sampling")
    logging.info("="*72)

    
    # Check lengths
    n_ids = len(ids)
    n_embed = len(embed)
    n_entropy = len(entropy)
    assert (n_ids == n_embed) and (n_ids == n_entropy)
    assert (docs is None) or (len(docs) == n_ids)
    logging.info("Doc count, original:\t{}".format(n_ids))
    logging.info("ID, embedding, and entropy counts match")
    logging.info("Input examples:")
    for i, (id_, h, v) in enumerate(zip(ids, entropy, embed)):
        logging.info("")
        logging.info("Sample {}".format(i))
        logging.info("id:     \t{}".format(id_))
        logging.info("entropy:\t{}".format(', '.join(['{}={:.0f}%'.format(det, h_) for det, h_ in h.items()])))
        logging.info('embed:  \t{}...{}'.format(v[:5], v[-5]))
        if i > 2:
            break

    if sample_count > n_ids:
        logging.warn("Desired sample count > available samples")
        logging.warn("Setting sample_count = {}".format(n_ids))
        sample_count = n_ids

    # Filter low entropy samples
    entropy_filt = lambda X, H, H_min: [x for x, h in zip(X, H) if h >= H_min]
    idx_keep = []
    for i, H in enumerate(entropy):
        h_sum = sum([h for evt_type, h in H.items()])
        if h_sum > entropy_min:
            idx_keep.append(i)
    keep = lambda X, I: [X[i] for i in I]
    ids = keep(ids, idx_keep)
    embed = keep(embed, idx_keep)
    entropy = keep(entropy, idx_keep)
    if docs is not None:
        docs = keep(docs, idx_keep)
    if prob is not None:
        prob = keep(prob, idx_keep)
    n_ids = len(ids)
    n_embed = len(embed)
    n_entropy = len(entropy)
    assert (n_ids == n_embed) and (n_ids == n_entropy)
    assert (docs is None) or (len(docs) == n_ids)
    logging.info("Minimum entropy:\t{}".format(entropy_min))
    logging.info("Doc count, after entropy filter:\t{}".format(n_ids))
    
    
    # Get event types
    event_types = list(entropy[0].keys())
    event_types_cycle = cycle(event_types)
    n_evt_typ = len(event_types)
    logging.info("Event type count:\t{}".format(n_evt_typ))
    logging.info("Event types:\t{}".format(event_types))

    # Convert to numpy arrays
    ids = np.array(ids)
    embed = np.array(embed)
    
    entropy = entropy_as_dict_array(entropy, entropy_type)

    docs = docs_as_array(docs, n_ids)
    prob = prob_as_dict_array(prob, n_ids)


    similarity = np.zeros_like(ids, dtype=np.float64)
    similarity_tot = np.zeros_like(ids)
    sim_j = np.zeros_like(ids)
    
    logging.info("")
    logging.info("Input as numpy array:")
    logging.info("ids:\t{}".format(ids.shape))
    for evt_typ, h in entropy.items():
        logging.info("entropy, {}:\t{}".format(evt_typ, h.shape))
    logging.info("embed:\t{}".format(embed.shape))
    logging.info("")
    
    # Select samples for annotation, one of the time
    sampled_ids = []        
    summary = []
    sampled_embeds = []
    logging.info("Selecting samples for batch")
    for i in range(sample_count):

        logging.info("")
        logging.info("Drawing sample:\t{}".format(i))
    
        # Get entropy type
        if entropy_type == H_TYPE_LOOP:
            evt_typ = next(event_types_cycle)
        
        # Sum entropy scores across determinants
        elif entropy_type == H_TYPE_SUM:
            evt_typ = 'all'
        else:
            raise ValueError("incorrect entropy_type:\t".format(entropy_type))
            
        # Entropy
        h = entropy[evt_typ]

        logging.info("Event type:\t{}".format(evt_typ))
            
        # Diversity score
        diversity = 1 - similarity
        if np.amin(diversity) <= 0:
            diversity = diversity.clip(min=div_min)
            logging.warn("At least 1 diversity socre <= 0")
            logging.warn("Clipping negative diversity scores to {}".format(div_min))
        
        
        # Document score
        diversity_mod = np.power(diversity, alpha)
        scores = h*diversity_mod

        # Index of sample with maximum score
        j = np.argmax(scores)
        
        # Get sample info
        id_j = ids[j].copy()
        embed_j = embed[j].copy()
        similarity_j = similarity[j].copy()
        diversity_j = diversity[j].copy()
        diversity_mod_j = diversity_mod[j].copy()
        h_j = h[j].copy()
        scores_j = scores[j].copy()
        docs_j = docs[j]
        
        # Add to sampled batch
        sampled_ids.append(id_j)
        
        # Create summary
        d = OrderedDict()
        d['iteration'] = i
        d[EVENT_TYPE] = evt_typ
        d['idx'] = j
        d['id'] = id_j
        d['similarity'] = similarity_j
        d['diversity'] = diversity_j
        d['diversity_mod'] = diversity_mod_j
        d['entropy'] = h_j
        d['score'] = scores_j
        d['text'] = docs_j

        summary.append(d)
        
        if ((n_examples is None) or (i < n_examples)) and (path is not None):
           
            # Plot summary
            fn = os.path.join(path, 'sample_{}_{}_{}.png'.format(i, id_j, evt_typ))
            plot_summary(diversity, diversity_mod, h, scores, fn)

            fn = os.path.join(path, 'sample_{}_{}_{}.csv'.format(i, id_j, evt_typ))
            tabular_summary(diversity, diversity_mod, prob, h, scores, docs, fn, n_sample=20)
            
 
        # Calculate similarity between selected sample and remaining samples
        sampled_embeds.append(embed_j)
        sim_j = np.matmul(embed, embed_j)
        
        # Accumulate similarity scores
        # Average similarity scores
        if sim_type == SIM_TYPE_AVG:
            similarity_tot = similarity_tot + sim_j
            similarity = similarity_tot/(i+1)
        # Maximum of similarity scores
        elif sim_type == SIM_TYPE_MAX:
            similarity = np.maximum(similarity, sim_j)
        else:
            raise ValueError("Invalid sim_type:\t{}".format(sim_type))
       
        # Remove selected sample from unlabeled set
        rm = lambda X, j: np.delete(X, (j), axis=0)            
        ids = rm(ids, j)
        embed = rm(embed, j)
        entropy = {et: rm(h, j) for et, h in entropy.items()}
        similarity = rm(similarity, j)
        similarity_tot = rm(similarity_tot, j)
        sim_j = rm(sim_j, j)
        docs = rm(docs, j)
        prob = {el: rm(p, j) for el, p in prob.items()}

        assert len(ids) == len(embed)
        for _, h in entropy.items():
            assert len(ids) == len(h)
        assert len(ids) == len(similarity)
        assert len(ids) == len(similarity_tot)
        assert len(ids) == len(sim_j)
        assert len(ids) == len(docs)
        for _, p in prob.items():
            assert len(ids) == len(p)
        
        
        logging.info("Index selected:\t{}".format(j))
        logging.info("Count in batch:  \t{}".format(len(sampled_ids)))
        logging.info("Count unselected:\t{}".format(len(ids)))
  
        # Stop iterating, if data empty
        if len(ids) == 0:
            logging.warn("No more samples. Unfull batch")
            break
            
            
  
    # Get remaining IDs (unsampled IDs)
    unsampled_ids = list(ids)
    assert len(unsampled_ids) + len(sampled_ids) == n_ids
    assert len(set(unsampled_ids) & set(sampled_ids)) == 0
    
    if path is not None:
        df = pd.DataFrame(summary)
        fn = os.path.join(path, 'summary.csv')
        df.to_csv(fn)
        
        for evt_typ in event_types:
            fn = os.path.join(path, 'summary_{}'.format(evt_typ))
            df_tmp = df[df[EVENT_TYPE] == evt_typ]
            if len(df_tmp) > 0:
                d = df_tmp['diversity']
                dm = df_tmp['diversity_mod']
                h = df_tmp['entropy']
                s = df_tmp['score']
                plot_summary(d, dm, h, s, fn, n_top=None, sort_scores=False)
                
        fn = os.path.join(path, 'similarity_distribution.png')
        plot_sim_combos(sampled_embeds, fn)
    
    return (sampled_ids, unsampled_ids)


def random_sample(X, size, seed=None):
    '''
    Sample X, drawing size without replacement
    
    Parameters
    ----------
    X: iterable, e.g. list of sentences, list of indices, etc.
    size: number of samples to draw randomly
    seed: seed for random number generator
    
    
    Returns
    -------
    Y: list of selected samples from X
    indices: indices of X in Y
    '''
    
    
    cnt = len(X)
    X_indices = list(range(cnt))
    
    # Random sample
    if (size is not None) and (cnt > size):

        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
        # Indices of sampled values
        Y_indices = np.random.choice(X_indices, \
                                   size = size, 
                                   replace = False, 
                                   p = None)
        Y_indices = np.sort(Y_indices)
    
        # Get sampled values based on indices
        Y = [X[i] for i in Y_indices] 
    
    # Pass through    
    else:
        Y = X[:]
        Y_indices = X_indices[:]
            
    return (Y, Y_indices)    

def random_split(X, size, seed=None):
    '''
    Randomly split X into two partitions
    '''
    
    cnt = len(X)
    X_indices = list(range(cnt))
    
    # Randomly select first partition
    Y, Y_indices = random_sample(X, size, seed=seed)
    
    # Identify sampled indices
    Y_indices_set = set(Y_indices)
    Z_indices = [i for i in X_indices if i not in Y_indices_set]
    
    # Check lengths
    assert len(Y_indices) + len(Z_indices) == cnt
    assert set(Y_indices).union(set(Z_indices)) == set(X_indices)
    
    # Get second partition samples
    Z = [X[i] for i in Z_indices]
    
    # Check lengths
    assert len(Y) + len(Z) == cnt
    
    return (Y, Y_indices, Z, Z_indices)




    

    
    