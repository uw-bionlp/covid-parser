import grpc
from proto.python.CovidParser_pb2 import Sentence, SentenceDetectionInput, SentenceDetectionOutput
from proto.python.CovidParser_pb2_grpc import OpenNLPStub

class OpenNLPClient():
    def __init__(self):
        self.name    = 'open-nlp'
        self.host    = '0.0.0.0'
        self.port    = '42400'
        self.open()

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        self.stub = OpenNLPStub(self.channel)
    
    def close(self):
        self.channel.close()

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