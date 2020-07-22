import torch
import gensim
import numpy as np
import os
import json
import pandas as pd
from collections import OrderedDict
from gensim.models import Word2Vec
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import time
import logging

from gensim.models.callbacks import CallbackAny2Vec
from gensim.test.utils import common_texts, get_tmpfile

from constants import *


def get_w2v_embed(model_dir, to_pytorch=False, freeze=True):
    
    '''
    Load pre-trained word2vec model and 
    create embedding matrix and token map
    '''

    
    fn = os.path.join(model_dir, MODEL_FILE)
    model = Word2Vec.load(fn)
    
    logging.info("Gensim model path:\t{}".format(fn))
    logging.info("Gensim model:\t{}".format(model))
    
    
    fn = os.path.join(model_dir, HYPERPARAMS_FILE)
    with open(fn,'r') as f:
        params = json.load(f)

    logging.info("Gensim model parameters path:\t{}".format(fn))        
    logging.info("Gensim model parameters:")
    for k, v in params.items():
        logging.info('\t{} = {}'.format(k, v))

    to_lower = params['to_lower']
    unk = params['unk']
    start_token = params['start_token']
    end_token = params['end_token']
           
    # Vocabulary
    vocab = list(model.wv.index2word)
    logging.info("\tVocab size:\t{}".format(len(vocab)))


    # Confirm unk in vocab
    assert unk in vocab, 'unk ({}) not in vocab'.format(unk)
    assert start_token in vocab, 'start_token ({}) not in vocab'.format(start_token)
    assert end_token in vocab, 'end_token ({}) not in vocab'.format(end_token)

    # Embedding matrix (vocab_size, embedding_dim)
    embed = np.array([model[v] for v in vocab])
    logging.info("\tEmbed matrix size:\t{}".format(embed.shape))
    
    # Map vocabulary to index
    vocab_to_index = OrderedDict()
    for i, k in enumerate(vocab):
        vocab_to_index[k] = i 
    
    # Vocab to ID function
    def vocab2id(tok, v2i=vocab_to_index, tl=to_lower, u=unk):

        # Convert to lowercase
        if tl:
            tok = tok.lower()            

        # ID for unk
        unk_id = v2i[u]

        # Map to ID
        id_ = v2i.get(tok, unk_id)

        return id_
    
    
    # Port embedding into pytorch tensor
    if to_pytorch:
    
        logging.info("\tConverting Gensim to pytorch embedding")
    
        # Modify format
        embed = embed.astype(np.float32)
    
        # Get word embeddings shape    
        (vocab_size, embed_dim) = tuple(embed.shape)
        logging.info("\t\tVocab size:\t{}".format(vocab_size))
        logging.info("\t\tEmbed size:\t{}".format(embed_dim))
    
        # Convert to embedding layer
        embed = torch.from_numpy(embed)
        embed = torch.nn.Embedding.from_pretrained( \
                                    embeddings = embed,
                                    freeze = freeze)
    
        logging.info("\t\ttorch embed:\t{}".format(embed))    
    return (vocab2id, embed)


class w2v_callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, path):
        

        
        self.epoch = 0
        self.loss_to_be_subed = 0
                
        self.out = []
        self.fn = os.path.join(path, 'w2v_loss.csv')
        self.fn_fig = os.path.join(path, 'w2v_loss.png')
        self.start_time = time.time()

        
    def on_epoch_end(self, model):
        
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        
        
        end_time = time.time()
        epoch_time = (end_time - self.start_time)/60
        self.start_time = end_time
        
        self.out.append((self.epoch, loss, loss_now, epoch_time))
        
    
        self.epoch += 1
        
        
        df = pd.DataFrame(self.out, columns=['Epoch', 'Loss total', 'Loss epoch', 'Time (m)'])
        df.to_csv(self.fn)
        
        ax = df.plot.bar(x='Epoch', y='Loss epoch', rot=0)
        ax.figure.savefig(self.fn_fig, quality=100, dpi=800)        
        plt.close('all')
        
        
        
    
'''
class word2vec(object):
    
    def __init__(self, \
        gensim_model, \
        hyperparams, \
        unk, \
        to_lower):

        self.gensim_model = gensim_model
        self.hyperparams = hyperparams
        self.unk = unk
        self.to_lower = to_lower
        

    def vocab(self):
        
        Get vocabulary from trained gensim_model
        
        return self.gensim_model.wv.index2word

    def word_embed(self):
       
        Create embedding matrix
        
        
        # Get embeddings
        embeddings = [self.gensim_model[v] for v in self.vocab()]

        # Reserve ID (index) 0
        # Pad to shift embeddings
        zeros = np.zeros_like(embeddings[0])
        embeddings.insert(0, zeros)
        
        return np.array(embeddings)
        
    def maps(self):
        
        #Token to index map, including unknown token cases
       # 
       # returns:
        #    word_map, index_map
        
        
        # Make sure vocab is list
        vocab = list(self.vocab())
        
        # Make sure unk is in vocab
        unk = self.unk
        assert unk in vocab, "Unk {} not in vocab".format(unk)

        # Map vocabulary to index
        # Reserve index 0
        vocab_to_index = {k:i+1 for i, k in enumerate(vocab)}

        # Map index to vocab
        index_to_vocab = {i:k for k, i in vocab_to_index.items()}         
        

        to_lower = self.to_lower
        
        # Vocab to ID function
        def vocab_to_id_fn(tok):

            # Convert to lowercase
            if to_lower:
                tok = tok.lower()            

            unk_id = vocab_to_index[unk]

            return vocab_to_index.get(tok, unk_id)
        
        # Token to ID function
        def id_to_vocab_fn(id_):
            return index_to_vocab[id_]
        
        return (vocab_to_id_fn, id_to_vocab_fn)



    def word_map(self):
        return self.maps()[0]
    
    
'''    
    
    
    