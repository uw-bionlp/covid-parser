from docker.base_extractor import BaseExtractor
from pytorch_models.multitask.model import MultitaskEstimator

class SDOHextractor(BaseExtractor):
    '''
    SDOH extraction model wrapper
    '''    
    def __init__(self, model_dir, word_embed_dir):
        '''
        Parameters
        ----------
        model_dir: directory with model files, as str
                   Should contain Pytorch state_dict.pt and hyperparams.json
        word_embed_dir: directory with Gensim word2vec model, as str        
        '''
                                
        super().__init__( \
            model_dir = model_dir,
            word_embed_dir = word_embed_dir,
            model_class = MultitaskEstimator, 
            ) 
