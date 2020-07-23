import grpc
from cli.utils import get_containers
from cli.constants import COVID
from proto.python.CovidParser_pb2 import PredictorInput, PredictorOutput
from proto.python.CovidParser_pb2_grpc import CovidStub
from copy import copy
import threading

def get_covid_containers():
    return [ container for key, container in get_containers().items() if COVID in container.name ]

class CovidPredictorChannelManager():
    def __init__(self, container):
        self.name = COVID
        self.host = container.host
        self.port = container.port

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')

    def close(self):
        self.channel.close()

    def generate_client(self, args):
        return CovidPredictorClient(self.channel, args)

class CovidPredictorClient():
    def __init__(self, channel, args):
        self.name           = COVID
        self.stub           = CovidStub(channel)
        self.channel        = channel

    def process(self, doc):
        response = self.stub.Predict(PredictorInput(id=doc.id, text=doc.text))
        return response

    def to_dict(self, response):
        output = { 'id': response.id, 'predictions': [] }
        for pred in response.predictions:
            prediction = {
                'entityType': pred.entity_type,
                'statusType': pred.status_type,
                'trigType': pred.trig_type,
                'type': pred.type,
            }
            for arg in pred.arguments:
                argument = {
                    'charIdxs': arg.char_idxs,
                    'label': arg.label,
                    'sentIdx': arg.sent_idx,
                    'text': arg.text,
                    'tokIdxs': arg.tok_idxs,
                    'tokens': arg.tokens,
                    'type': arg.type
                }
                prediction.arguments.append(argument)
            output.predictions.appned(prediction)
        return output

    def merge(self, base_json, client_json):
        base_json['covid'] = client_json
        return base_json
