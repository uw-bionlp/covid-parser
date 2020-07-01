import grpc
from cli.utils import get_containers
from cli.constants import METAMAP
from proto.python.CovidParser_pb2 import MetaMapInput, MetaMapOutput
from proto.python.CovidParser_pb2_grpc import MetaMapStub
from copy import copy
import threading

lck = threading.Lock()
sent_cache = {}

def get_metamap_containers():
    return [ container for key, container in get_containers().items() if METAMAP in container.name ]

class MetaMapChannelManager():
    def __init__(self, container):
        self.name = METAMAP
        self.host = container.host
        self.port = container.port

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')

    def close(self):
        self.channel.close()

    def generate_client(self, args):
        return MetaMapClient(self.channel, args)

class MetaMapClient():
    def __init__(self, channel, args):
        self.name           = METAMAP
        self.stub           = MetaMapStub(channel)
        self.channel        = channel
        self.semantic_types = args.metamap_semantic_types

    def process(self, doc):

        # Check if each sentence seen before
        global sent_cache
        global lck
        lck.acquire()
        cache = []
        for sent in doc.sentences:
            h = hash(sent.text)
            cached = h in sent_cache
            if cached:
                cpy = copy(sent_cache[h])
                cpy.id = sent.id
                cache.append(cpy)
                del doc.sentences[sent.id]
        lck.release()

        response = self.stub.ExtractNamedEntities(MetaMapInput(id=doc.id, sentences=doc.sentences, semantic_types=self.semantic_types))

        # Cache newly run sentences
        lck.acquire()
        for sent in response.sentences:
            h = hash(sent.text)
            sent_cache[h] = sent
        global lck
        
        # Insert previously cached sentences back in
        for sent in cache:
            response.sentences.insert(sent.id, sent)

        return response

    def to_dict(self, response):
        output = { 'id': response.id, 'sentences': [], 'errors': [ err for err in response.errors] }
        for sent in response.sentences:
            sentence = { 
                'id': sent.id, 
                'text': sent.text, 
                'concepts': [], 
                'beginCharIndex': sent.begin_char_index,
                'endCharIndex': sent.end_char_index,
            }
            for con in sent.concepts:
                concept = {
                    'beginSentenceCharIndex': con.begin_sent_char_index,
                    'endSentenceCharIndex': con.end_sent_char_index,
                    'beginDocumentCharIndex': con.begin_doc_char_index,
                    'endDocumentCharIndex': con.end_doc_char_index,
                    'cui': con.cui,
                    'semanticTypes': [ st for st in con.semantic_types ],
                    'sourcePhrase': con.source_phrase,
                    'conceptName': con.concept_name,
                    'prediction': con.prediction
                }
                sentence['concepts'].append(concept)
            output['sentences'].append(sentence)
        return output

    def merge(self, base_json, client_json):
        if len(base_json['sentences']) != len(client_json['sentences']):
            print(f"{self.name} sentence count doesn't match input sentence count! Skipping result merge.")

        base_json['errors'] = client_json['errors']
        for i,sentence in enumerate(base_json['sentences']):
            sentence[self.name] = client_json['sentences'][i]['concepts']

        return base_json

    def to_brat(self, client_json):
        t = 1
        brat_rows = []
        for sentence in client_json['sentences']:
            for con in sentence['concepts']:
                row = f"T{t}    {con['prediction']} {con['beginDocumentCharIndex']} {con['endDocumentCharIndex']}    {con['sourcePhrase']}"
                brat_rows.append(row)
                t += 1
        return brat_rows

