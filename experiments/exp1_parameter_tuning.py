import numpy as np
import json
import subprocess
import os
from multiprocessing import Process
import argparse


def load_json(json_path):
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out

def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=True)
    args = parser.parse_args()

    # Data Configuration
    json_data_path = '../config/base_data.json'
    json_data = load_json(json_data_path)

    # Network Configuration
    json_network_path = '../config/base_network.json'
    json_network = load_json(json_network_path)

    # Meta Configuration
    json_meta_path = '../config/base_meta.json'
    json_meta = load_json(json_meta_path)

    # Setup Configuration for Each Experiments
    if args.exp == 0:
        server = 'nipa'
        save_dir = '/data/sung/checkpoint/low_il'
        data_dir = '/data/sung/dataset/low-illumination-dataset'

        exp_name = 'init'
        start = 0
        comb_list = []

        num_per_gpu = 1
        gpus = ['0,1,2']
        train_list = ['robust']
        optimizer_type = 'sgd'

        # Resume Option
        resume = False

        ix = 0
        for tr in train_list:
            comb_list.append([tr, ix])
            ix += 1

    else:
        raise('Select Proper Experiment Number')

    comb_list = comb_list * num_per_gpu
    comb_list = [comb + [index] for index, comb in enumerate(comb_list)]

    arr = np.array_split(comb_list, len(gpus))
    arr_dict = {}
    for ix in range(len(gpus)):
        arr_dict[ix] = arr[ix]

    def tr_gpu(comb, ix):
        comb = comb[ix]
        for i, comb_ix in enumerate(comb):
            exp_num = start + int(comb_ix[-1])
            os.makedirs(os.path.join(save_dir, exp_name, str(exp_num)), exist_ok=True)

            gpu = gpus[ix]

            # Modify the data configuration
            json_data['data_dir'] = str(data_dir)
            save_json(json_data, os.path.join(save_dir, exp_name, str(exp_num), 'data.json'))


            # Modify the network configuration
            save_json(json_network, os.path.join(save_dir, exp_name, str(exp_num), 'network.json'))


            # Modify the train configuration
            train_type = str(comb_ix[0])
            json_train_path = '../config/base_train_%s.json' %train_type
            json_train = load_json(json_train_path)

            json_train['resume'] = resume
            json_train['gpu'] = str(gpu)
            save_json(json_train, os.path.join(save_dir, exp_name, str(exp_num), 'train.json'))


            # Modify the optimizer configuration
            json_optim_path = '../config/base_optim_%s.json' %optimizer_type
            json_optim = load_json(json_optim_path)
            save_json(json_optim, os.path.join(save_dir, exp_name, str(exp_num), 'optim.json'))


            # Modify the meta configuration
            json_meta['server'] = str(server)
            json_meta['save_dir'] = str(save_dir)
            save_json(json_meta, os.path.join(save_dir, exp_name, str(exp_num), 'meta.json'))


            # Run !
            script = 'python ../main.py --save_dir %s --exp_name %s --exp_num %d' % (save_dir, exp_name, exp_num)
            subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)