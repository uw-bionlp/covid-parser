#!/usr/bin/env python3

import os
import grpc
import json
from pathlib import Path
from argparse import ArgumentParser
from time import localtime, strftime

from cli.utils import mm_response_to_dict

from proto.OpenNLP.python.OpenNLP_pb2 import Sentence, SentenceDetectionInput, SentenceDetectionOutput
from proto.OpenNLP.python.OpenNLP_pb2_grpc import OpenNLPStub
from proto.MetaMap.python.MetaMap_pb2 import MetaMapInput, MetaMapOutput
from proto.MetaMap.python.MetaMap_pb2_grpc import MetaMapStub

OPEN_NLP_PORT = '42400'
METAMAP_PORT = '42402'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file_or_dir', help='Absolute or relative path to the file or directory to parse.')
    parser.add_argument('-o', '--output', help='Directory to write output to. Defaults to /output/<now>/', default=f'{os.getcwd()}{os.path.sep}output{os.path.sep}{strftime("%Y%m%d-%H%M%S")}')
    args = parser.parse_args()

    args.file_or_dir = os.path.abspath(args.file_or_dir)
    path_valid = os.path.exists(args.file_or_dir)
    is_file = os.path.isfile(args.file_or_dir)
    
    return args.file_or_dir, args.output, path_valid, is_file


def parse_with_metamap(mm_stub, id, sentences, semantic_types):
    """ Parse the document with MetaMap and Assertion Classifier.

    Arguments:
    - mm_stub        -- MetaMap gRPC client stub
    - id             -- ID of the document
    - sentences      -- Split sentences
    - semantic_types -- UMLS types to include
    """
    
    response = mm_stub.ExtractNamedEntities(MetaMapInput(id=id, sentences=sentences, semantic_types=semantic_types))
    return mm_response_to_dict(response)

def split_sentences(open_nlp_stub, id, text):
    """ Split text into multiple sentences """

    return open_nlp_stub.DetectSentences(SentenceDetectionInput(id=id, text=text))


def write_output(output, output_path):
    """ Write output to directory in pretty-printed JSON. """

    filename = output['id']
    with open(f'{output_path}{os.path.sep}{filename}.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def get_processors():
    METAMAP = 'metamap'
    stubs = {}
    processors = []

    # MetaMap
    stubs[METAMAP] = grpc.insecure_channel(f'0.0.0.0:{METAMAP_PORT}')
    processors.append(lambda stubs, doc_id, sentences: parse_with_metamap(stubs[METAMAP], doc_id, sentences, []))

    return stubs, processors

def process_file(open_nlp_stub, filepath, stubs, processors):

    results = []
    with open(filepath, 'r') as f:
        text = f.read()
        doc_id = Path(filepath).stem

    # Get sentences. 
    sentences = split_sentences(open_nlp_stub, doc_id, text)

    for processor in processors:
        results = processor(stubs, doc_id, sentences)
    x=1


def run():
    """ Run the client. """

    # Parse args, bail if invalid.
    file_or_dir, output_path, path_valid, is_file = parse_args()
    if not path_valid:
        print(f"The file or directory '{file_or_dir}' could not be found!")
        return

    # Make output directory.
    # Path(output_path).mkdir(parents=True, exist_ok=True)

    # Get stubs and processors.
    stubs, processors = get_processors()

    # Get OpenNLP stub.
    with grpc.insecure_channel(f'0.0.0.0:{OPEN_NLP_PORT}') as channel:
        open_nlp_stub = OpenNLPStub(channel)

    # If a file, parse only that.
        if is_file:
            print(f"Parsing file '{file_or_dir}'...")
            process_file(open_nlp_stub, file_or_dir, stubs, processors)

        # Else parse all .txt files in the directory.
        else:
            files = [ f'{file_or_dir}{os.path.sep}{f}' for f in os.listdir(file_or_dir) if Path(f).suffix == '.txt' ]
            print(f"Found {len(files)} text files in '{file_or_dir}'...")
            for i,f in enumerate(files, 1):
                print(f"Parsing file {i}...")
                process_file(open_nlp_stub, file_or_dir, stubs, processors)
            
    print(f"All done! Results written to '{output_path}'") 


if __name__ == '__main__':
    run()