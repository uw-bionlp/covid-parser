#!/usr/bin/env python3

import sys
import time
import subprocess
from cli.utils import get_container_runtime, get_env_vars, get_images, get_containers, run_shell_cmd
from cli.constants import *

runtime = get_container_runtime()

def build(img_name, path):
    cmd = f'{runtime} build -t {img_name} -f {path}/Dockerfile {path}/'
    sys.stdout.write(f'{cmd}\n')
    run_shell_cmd(cmd)

def run(name, img_name, bind_port_to):
    cmd = f'{runtime} run --name={name} -p {bind_port_to}:8080 {img_name} -d'
    sys.stdout.write(f'{cmd}\n')
    run_shell_cmd(cmd, True)

def wait_till_up(name):
    wait_cnt = 0
    wait_seconds = 2
    time.sleep(wait_seconds)
    up = False
    while not up:
        container = get_containers().get(name)
        up = container.up if container else False
        if not up:
            time.sleep(wait_seconds)
            wait_cnt += 1
            if wait_cnt >= 10:
                sys.stdout.write(f'The container {name} is still not up after {wait_cnt * 10} seconds. Is something wrong?\n')
                sys.exit()

def main():
    images = [ img for key, img in get_images().items() ]
    containers = [ container for key, container in get_containers().items() ]
    env_vars = get_env_vars()
    mmlite_inst_cnt = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1

    # Assertion Classifier
    # cont_name = f'{APP_NAME}_{ASSERTION_CLASSIFIER}_1'
    # img_name = f'{APP_NAME}_{ASSERTION_CLASSIFIER}'
    # port = env_vars[ENV_ASRTCLA_PORT]

    # if not images.get(img_name):
    #     build(img_name, ASSERTION_CLASSIFIER)
    # if not any([ x for x in containers if cont_name in x.name ]):
    #     run(cont_name, img_name, port)
    #     wait_till_up(cont_name)

    # Open-NLP
    cont_name = f'{APP_NAME}_{OPEN_NLP}_1'
    img_name = f'{APP_NAME}_{OPEN_NLP}'
    port = env_vars[ENV_OPENNLP_PORT]

    if not any([ x for x in images if img_name in x.name ]):
        build(img_name, OPEN_NLP)
    if not any([ x for x in containers if cont_name in x.name ]):
        run(cont_name, img_name, port)
        wait_till_up(cont_name)

    # MetaMap
    img_name = f'{APP_NAME}_{METAMAP}'
    port = int(env_vars[ENV_METAMAP_PORT])

    if not any([ x for x in images if img_name in x.name ]):
        build(img_name, METAMAP)
    
    for i in range(mmlite_inst_cnt):
        cont_name = f'{APP_NAME}_{METAMAP}_{i+1}'
        if not any([ x for x in containers if cont_name in x.name ]):
            run(cont_name, img_name, port)
            wait_till_up(cont_name)
        port += 1

    # Clear dangling PIDs
    #cmd = "ps x | grep './up.sh' | awk '{print $1}' | xargs kill -9"
    #run_shell_cmd(cmd)
    pids = [ l.strip().split(' ')[0] for l in str(run_shell_cmd('ps x')[0]).split('\\n') if './up.sh' in l and 'python3' in l ]
    for p in pids:
        print(f'kill -9 {p}')
        run_shell_cmd(f'kill -9 {p}')

if __name__ == '__main__':
    main()