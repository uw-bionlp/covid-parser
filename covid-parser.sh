#!/usr/bin/env python3

import os
import sys
import grpc
import json
import queue
from pathlib import Path
from argparse import ArgumentParser
from time import localtime, strftime
from multiprocessing.dummy import Process, Queue, current_process
import threading

from cli.clients.metamap import MetaMapChannelManager, get_metamap_containers
from cli.clients.opennlp import OpenNLPChannelManager
from cli.clients.assertion_classifier import AssertionClassifierChannelManager

""" Globals """
lck = threading.Lock()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('file_or_dir', help='Absolute or relative path to the file or directory to parse.')
    parser.add_argument('-o', '--output_path', help='Directory to write output to. Defaults to /output/<current_time>/')
    parser.add_argument('-t', '--threads', help='Number of threads with which to execute processing in parallel. Defaults to one.', default=1, type=int)
    parser.add_argument('--metamap', help='Whether to parse with MetaMap or not. Defaults to false.', default=False, dest='metamap', action='store_true')
    parser.add_argument('--metamap_semantic_types', help="MetaMap semantic types to include (eg, 'sosy', 'fndg'). Defaults to all.", nargs='+')
    parser.add_argument('--brat', help='Output BRAT-format annotation files, in addition to JSON.', default=False, dest='brat', action='store_true')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit()

    args.file_or_dir = os.path.abspath(args.file_or_dir)
    if not args.output_path:
        args.output_path = f'{os.getcwd()}{os.path.sep}output{os.path.sep}{Path(args.file_or_dir).stem}_{strftime("%Y%m%d-%H%M%S")}'
    
    return args


def write_output(output, output_path):
    """ Write output to directory in pretty-printed JSON. """

    filename = output['id']
    with open(f'{output_path}{os.path.sep}{filename}.json', 'w') as f:
        json.dump(output, f, ensure_ascii=False, indent=4)

def get_channels(args):
    channels = []

    # If Metamap
    if args.metamap:
        metamap_containers = get_metamap_containers()
        mm_cnt = len(metamap_containers)
        if mm_cnt < args.threads:
            cnt = f'only {mm_cnt}' if mm_cnt > 0 else 'no'
            sys.stdout.write(f'Error: {args.threads} MetaMap threads requested but {cnt} available. Exiting...\n')
            sys.exit()
        return [ MetaMapChannelManager(container) for container in metamap_containers[:args.threads] ]

    # If BRAT
    if args.brat:
        for channel in channels:
            Path(f'{args.output_path}{os.path.sep}brat_{channel.name}').mkdir(parents=True, exist_ok=True)
    
    return channels

def process_doc(filepath, opennlp_client, clients, output_path):
    with open(filepath, 'r') as f:
        text = f.read()
        doc_id = Path(filepath).stem

    # Get base sentences, locking opennlp between threads.
    global lck
    lck.acquire()
    doc = opennlp_client.process(doc_id, text)
    json = opennlp_client.to_dict(doc)
    lck.release()

    # For each gRPC client, process file.
    for client in clients:
        response = client.process(doc)
        client_json = client.to_dict(response)
        json = client.merge(json, client_json)

    write_output(json, output_path)

def do_subprocess(remaining, completed, args, opennlp_channel, channels, thread_idx):
    
    # Get gRPC clients.
    clients = [ channel.generate_client(args) for channel in channels ]
    opennlp_client = opennlp_channel.generate_client()
    errored, succeeded = 0, 1

    while True:
        try:
            doc = remaining.get_nowait()
        except queue.Empty:
            break
        sys.stdout.write(f'Processing document "{doc}"... by thread {thread_idx}\n')
        try:
            process_doc(doc, opennlp_client, clients, args.output_path)
            completed.put(doc)
            succeeded += 1
        except Exception as ex:
            errored += 1

            # If only run a handful of times, continue trying.
            if errored+succeeded < 5:
                remaining.put(doc)
                sys.stdout.write(f'"{doc}" failed, retrying...\n')
                continue

            # If under threshold (and thus likely going smoothly), retry.
            pct = errored / succeeded
            if pct < 0.2:
                remaining.put(doc)
            else:
                sys.stdout.write(f'{round(pct * 100,1)}% of documents failed, not retrying "{doc}"...\n')
                sys.stdout.write(f'Error: {ex}')
            
    return True

def main():
    """ Run the client. """

    # Parse args, bail if invalid.
    args = parse_args()
    if not os.path.exists(args.file_or_dir):
        sys.stdout.write(f"The file or directory '{args.file_or_dir}' could not be found!\n")
        return

    # Make output directory.
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Load documents
    if os.path.isfile(args.file_or_dir):
        files = [ args.file_or_dir ]
    else:
        files = [ f'{args.file_or_dir}{os.path.sep}{f}' for f in os.listdir(args.file_or_dir) if Path(f).suffix == '.txt' ]
        sys.stdout.write(f"Found {len(files)} text files in '{args.file_or_dir}'...\n")

    # Get and open gRPC channels.
    opennlp_channel = OpenNLPChannelManager()
    opennlp_channel.open()
    channels = get_channels(args)
    for channel in channels:
        channel.open()

    # Process multithread.
    remaining = Queue()
    completed = Queue()

    for f in files:
        remaining.put(f)

    processes = []
    if args.metamap:
        for w,channel in enumerate(channels):
            p = Process(target=do_subprocess, args=(remaining, completed, args, opennlp_channel, [ channel ], w))
            processes.append(p)
            p.start()
    else:   
        for w in range(args.threads):
            p = Process(target=do_subprocess, args=(remaining, completed, args, opennlp_channel, channels, w))
            processes.append(p)
            p.start()
    for p in processes:
        p.join()

    '''
    threads = []
    if args.metamap:
        for w,channel in enumerate(channels):
            t = threading.Thread(target=do_subprocess, args=(remaining, completed, args, opennlp_channel, [ channel ], w))
            threads.append(t)
            t.start()
    else:   
        for w in range(args.threads):
            t = threading.Thread(target=do_subprocess, args=(remaining, completed, args, opennlp_channel, channels, w))
            threads.append(t)
            t.start()
    for t in threads:
        t.join()
    '''

    # Close all gRPC channels.
    opennlp_channel.close()
    for channel in channels:
        channel.close()
            
    print(f"All done! Results written to '{args.output_path}'\n") 


if __name__ == '__main__':
    main()