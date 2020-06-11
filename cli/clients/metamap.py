import grpc
from proto.python.CovidParser_pb2 import MetaMapInput, MetaMapOutput
from proto.python.CovidParser_pb2_grpc import MetaMapStub

class MetaMapChannel():
    def __init__(self):
        self.name    = 'metamap'
        self.host    = '0.0.0.0'
        self.port    = '42402'

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')

    def close(self):
        self.channel.close()

    def generate_client(self, args):
        return MetaMapClient(self.channel, args)

class MetaMapClient():
    def __init__(self, channel, args):
        self.name = 'metamap'
        self.stub = MetaMapStub(channel)
        self.semantic_types = args.metamap_semantic_types

    def process(self, doc):
        response = self.stub.ExtractNamedEntities(MetaMapInput(id=doc.id, sentences=doc.sentences, semantic_types=self.semantic_types))
        return response

    def to_dict(self, response):
        output = { 'id': response.id, 'sentences': [] }
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