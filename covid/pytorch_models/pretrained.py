
import torch
import os
import json
import joblib
import logging

from constants import *


def load_pretrained(model_class, model_dir, word_embed_dir, param_map=None):
    '''
    Load pretrained Pytorch model
    '''
    
    
    logging.info('')
    logging.info('-'*72)
    logging.info('Loading pre-trained model from:\t{}'.format(model_dir))
    logging.info('-'*72)
    
    # Load hyper parameters
    fn = os.path.join(model_dir, HYPERPARAMS_FILE)
    logging.info("\tHyperparameters file:\t{}".format(fn))
    hyperparams = json.load(open(fn,'r'))
    
    # Print hyper parameters
    logging.info("\tHyperparameters loaded:")
    for param, val in hyperparams.items():
        logging.info("\t\t{}:\t{}".format(param, val))
    logging.info('')

    # Map parameters
    if param_map is not None:
        logging.info('\tMapping hyperparameters:')
        for name, val in param_map.items():
            logging.info('\t\t{}: orig={},\tnew={}'.format(name, hyperparams[name], val))
            hyperparams[name] = val
        
    # Changed(ndobb) - Override word2vec dir
    hyperparams['word_embed_dir'] = word_embed_dir

    # Load saved estimator
    fn = os.path.join(model_dir, STATE_DICT)
    logging.info("\tState dict file:\t{}".format(fn))
    state_dict = torch.load(fn, map_location=lambda storage, loc: storage)
    logging.info("\tState dict loaded")
    
    # Instantiate model
    model = model_class(**hyperparams)
    model.load_state_dict(state_dict)   
    logging.info("\tModel instantiated and state dict loaded")
    logging.info('')
    
    return (model, hyperparams)
