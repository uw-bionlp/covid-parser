#!/usr/bin/env python3

import os
import sys
import grpc
import json
from pathlib import Path
from argparse import ArgumentParser
from time import localtime, strftime

from cli.clients.metamap import MetaMapClient
from cli.clients.opennlp import OpenNLPClient


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file_or_dir', help='Absolute or relative path to the file or directory to parse.')
    parser.add_argument('-o', '--output_path', help='Directory to write output to. Defaults to /output/<now>/')
    parser.add_argument('--metamap', help='Whether to parse with MetaMap or not. Defaults to false.', default=False, dest='metamap', action='store_true')
    parser.add_argument('--metamap_semantic_labels', help='MetaMap semantic labels to include. Defaults to all.', nargs='+')
    parser.add_argument('--brat', help='Output BRAT-format annotation files, in addition to JSON.', default=False)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    args.file_or_dir = os.path.abspath(args.file_or_dir)
    if not args.output_path:
        args.output_path = f'{os.getcwd()}{os.path.sep}output{os.path.sep}{Path(args.file_or_dir).stem}_{strftime("%Y%m%d-%H%M%S")}'
    
    return args


def write_output(output, output_path):
    """ Write output to directory in pretty-printed JSON. """

    filename = output['id']
    with open(f'{output_path}{os.path.sep}{filename}.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)


def get_clients(args):
    clients = []
    if args.metamap:
        clients.append(MetaMapClient(args))
    
    return clients


def process_file(opennlp_client, clients, filepath):
    with open(filepath, 'r') as f:
        text = f.read()
        doc_id = Path(filepath).stem

    # Get base sentences.
    doc = opennlp_client.process(doc_id, text)
    json = opennlp_client.to_dict(doc)

    # For each gRPC client, process file.
    for client in clients:
        response = client.process(doc)
        client_json = client.to_dict(response)
        json = client.merge(json, client_json)

    return json

def run():
    """ Run the client. """

    # Parse args, bail if invalid.
    args = parse_args()
    if not os.path.exists(args.file_or_dir):
        print(f"The file or directory '{args.file_or_dir}' could not be found!")
        return

    # Make output directory.
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Get gRPC clients.
    clients = get_clients(args)
    opennlp_client = OpenNLPClient()

    if os.path.isfile(args.file_or_dir):
        files = [ args.file_or_dir ]
    else:
        files = [ f'{args.file_or_dir}{os.path.sep}{f}' for f in os.listdir(args.file_or_dir) if Path(f).suffix == '.txt' ]
        print(f"Found {len(files)} text files in '{args.file_or_dir}'...")
    
    # Process each file.
    for i,f in enumerate(files, 1):
        print(f"Parsing file {i}...")
        results = process_file(opennlp_client, clients, f)
        write_output(results, args.output_path)

    # Close all gRPC clients.
    opennlp_client.close()
    for client in clients:
        client.close()
            
    print(f"All done! Results written to '{args.output_path}'") 


if __name__ == '__main__':
    run()