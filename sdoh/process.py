import os
import json

from docker.extractor import Extractor
from grpc_server.CovidParser_pb2 import PredictionOutput, PredictionEvent, PredictionEventArgument

class DocumentProcessor():

    def __init__(self):
        self.model_dir      = os.path.join('model','sdoh')
        self.word_embed_dir = os.path.join('model','word2vec')
        self.extractor      = Extractor('sdoh', self.model_dir, self.word_embed_dir)

    def predict(self, text):
        prediction = self.extractor.predict(text, -1, 'obj')

        # Return Protobuf Result object.
        result = PredictionOutput()
        for ev in prediction:
            for e in ev:
                pred_ev = PredictionEvent()
                pred_ev.entity_type = val_else_empty_str(e.entity_type)
                pred_ev.status_type = val_else_empty_str(e.status_type)
                pred_ev.trig_type = val_else_empty_str(e.trig_type)
                pred_ev.type = val_else_empty_str(e.type_)
                for arg in e.arguments:
                    pred_arg = PredictionEventArgument()
                    pred_arg.char_idxs.extend(list(arg.char_idxs) if arg.char_idxs else [])
                    pred_arg.label = val_else_empty_str(arg.label)
                    pred_arg.sent_idx = val_else_default_int(arg.sent_idx)
                    pred_arg.text = val_else_empty_str(arg.text)
                    pred_arg.tok_idxs.extend(list(arg.tok_idxs) if arg.tok_idxs else [])
                    pred_arg.tokens.extend(val_else_empty_list(arg.tokens))
                    pred_arg.type = val_else_empty_str(arg.type_)
                    pred_ev.arguments.append(pred_arg)
                result.predictions.append(pred_ev)

        return result

def val_else_empty_list(val):
    if val: return val
    return []

def val_else_empty_str(val):
    if val: return val
    return ''

def val_else_default_int(val):
    if val != None: return val
    return -1