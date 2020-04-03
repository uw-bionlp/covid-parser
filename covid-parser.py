#!/usr/bin/env python3

import os
import grpc
import json
from pathlib import Path
from argparse import ArgumentParser
from time import localtime, strftime

from pyclient.utils import mm_response_to_dict

from proto.MetaMap.python.MetaMap_pb2 import MetaMapInput, MetaMapOutput
from proto.MetaMap.python.MetaMap_pb2_grpc import MetaMapStub


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file_or_dir', help='Absolute or relative path to the file or directory to parse.')
    parser.add_argument('-o', '--output', help='Directory to write output to. Defaults to /output/<now>/', default=f'{os.getcwd()}{os.path.sep}output{os.path.sep}{strftime("%Y%m%d-%H%M%S")}')
    args = parser.parse_args()

    args.file_or_dir = os.path.abspath(args.file_or_dir)
    path_valid = os.path.exists(args.file_or_dir)
    is_file = os.path.isfile(args.file_or_dir)
    
    return args.file_or_dir, args.output, path_valid, is_file


def parse_with_metamap(mmstub, path, output_path):
    """ Parse the document with MetaMap and Assertion Classifier.

    Arguments:
    - mmstub      -- MetaMap gRPC client stub
    - path        -- Path to the file to read from
    - output_path -- Directory to write results to
    """
    
    with open(path) as f:
        text = f.read()
    response = mmstub.ExtractNamedEntities(MetaMapInput(id=Path(path).stem, text=text))
    output = mm_response_to_dict(response)
    write_output(output, output_path)
    

def write_output(output, output_path):
    """ Write output to directory in pretty-printed JSON. """

    filename = output['id']
    with open(f'{output_path}{os.path.sep}{filename}.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def run():
    """ Run the client. """

    # Parse args, bail if invalid.
    file_or_dir, output_path, path_valid, is_file = parse_args()
    if not path_valid:
        print(f"The file or directory '{file_or_dir}' could not be found!")
        return

    # Open gRPC channel.
    with grpc.insecure_channel('0.0.0.0:5000') as channel:
        mmstub = MetaMapStub(channel)

        # Make output directory if it doesn't exist.
        Path(output_path).mkdir(parents=True, exist_ok=True)

        # If a file, parse only that.
        if is_file:
            print(f"Parsing file '{file_or_dir}'...")
            parse_with_metamap(mmstub, file_or_dir, output_path)

        # Else parse all .txt files in the directory.
        else:
            files = [ f'{file_or_dir}{os.path.sep}{f}' for f in os.listdir(file_or_dir) if Path(f).suffix == '.txt' ]
            print(f"Found {len(files)} text files in '{file_or_dir}'...")
            for i,f in enumerate(files, 1):
                print(f"Parsing file {i}...")
                parse_with_metamap(mmstub, f, output_path)
            
    print(f"All done! Results written to '{output_path}'") 


if __name__ == '__main__':
    run()