import grpc
from proto.python.CovidParser_pb2 import MetaMapInput, MetaMapOutput
from proto.python.CovidParser_pb2_grpc import MetaMapStub

class MetaMapClient():
    def __init__(self, args):
        self.name            = 'metamap'
        self.host            = '0.0.0.0'
        self.port            = '42402'
        self.semantic_labels = args.metamap_semantic_labels
        self.open()

    def open(self):
        self.channel = grpc.insecure_channel(f'{self.host}:{self.port}')
        self.stub = MetaMapStub(self.channel)
    
    def close(self):
        self.channel.close()

    def process(self, doc):
        response = self.stub.ExtractNamedEntities(MetaMapInput(id=doc.id, sentences=doc.sentences, semantic_types=self.semantic_labels))
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
                    'semanticTypes': con.semantic_types,
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