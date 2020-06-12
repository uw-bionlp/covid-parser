import grpc
from cli.utils import get_containers, get_env_var
from cli.constants import OPEN_NLP, ENV_OPENNLP_PORT
from proto.python.CovidParser_pb2 import Sentence, SentenceDetectionInput, SentenceDetectionOutput
from proto.python.CovidParser_pb2_grpc import OpenNLPStub

def get_opennlp_containers():
    return [ container for key, container in get_containers().items() if OPEN_NLP in container.name ]

class OpenNLPChannelManager():
    def __init__(self):
        self.name = OPEN_NLP
        self.host = '0.0.0.0'
        self.port = get_env_var(ENV_OPENNLP_PORT)

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')

    def close(self):
        self.channel.close()

    def generate_client(self):
        return OpenNLPClient(self.channel)

class OpenNLPClient():
    def __init__(self, channel):
        self.name    = OPEN_NLP
        self.stub    = OpenNLPStub(channel)
        self.channel = channel

    def process(self, doc_id, text):
        response = self.stub.DetectSentences(SentenceDetectionInput(id=doc_id, text=text))
        for error in response.errors:
            print(error)
        return response

    def to_dict(self, response):
        output = { 'id': response.id, 'sentences': [], 'text': response.text }
        for sent in response.sentences:
            sentence = { 
                'id': sent.id, 
                'text': sent.text, 
                'beginCharIndex': sent.begin_char_index,
                'endCharIndex': sent.end_char_index,
            }
            output['sentences'].append(sentence)
        return output