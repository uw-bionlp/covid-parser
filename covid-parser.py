import os
import grpc
import json
from pathlib import Path
from argparse import ArgumentParser
from time import localtime, strftime

from proto.MetaMap.python.MetaMap_pb2 import MetaMapInput, MetaMapOutput
from proto.MetaMap.python.MetaMap_pb2_grpc import MetaMapStub

today = strftime("%Y%m%d-070605")
outpath = f'{os.getcwd()}{os.path.sep}output{os.path.sep}{today}'

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('-f', '--file', help='Absolute or relative path to the file to parse. If both --directory and --file are supplied, --file is ignored.')
    parser.add_argument('-d', '--dir', help='Absolute or relative path to the directory of files to parse.')
    args = parser.parse_args()

    if args.file:
        args.file = os.path.abspath(args.file)
    if args.dir:
        args.dir = os.path.abspath(args.dir)
    if not args.file and not args.dir:
        raise ValueError('You must supply either a file (-f) or directory (-d)!')
    
    return args.file, args.dir

def parse_with_metamap(mmstub, path):
    with open(path) as f:
        text = f.read()
    response = mmstub.ExtractNamedEntities(MetaMapInput(id=Path(path).stem, text=text))
    output = mm_response_to_dict(response)
    write_output(output)
    
def write_output(output):
    filename = output['id']
    with open(f'{outpath}{os.path.sep}{filename}.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def run():
    filename, directory = parse_args()
    with grpc.insecure_channel('0.0.0.0:5000') as channel:
        mmstub = MetaMapStub(channel)
        Path(outpath).mkdir(parents=True, exist_ok=True)

        if directory:
            print(f"Parsing all .txt files in '{directory}'...")
            files = [ f'{directory}{os.path.sep}{f}' for f in os.listdir(directory) if Path(f).suffix == '.txt' ]
            for i,f in enumerate(files, 1):
                print(f"Parsing file {i} '{f}'...")
                parse_with_metamap(mmstub, f)
        else:
            print(f"Parsing file '{filename}'...")
            parse_with_metamap(mmstub, filename)
    print('All done!')

def mm_response_to_dict(response):
    output = { 'id': response.id, 'text': response.text, 'sentences': [] }
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
                'semanticLabel': con.semantic_label,
                'sourcePhrase': con.source_phrase,
                'conceptName': con.concept_name,
                'prediction': con.prediction
            }
            sentence['concepts'].append(concept)
        output['sentences'].append(sentence)
    return output                 

if __name__ == '__main__':
    run()