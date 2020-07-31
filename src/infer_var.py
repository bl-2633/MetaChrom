"""
Inference script for variant effect
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import os
import torch
import argparse
import numpy as np
from model import MetaChrom, data_loader
from torch.utils import data as D

def var_inference(model, loader, device):
    result = dict()
    torch.cuda.set_device(device)
    model.cuda().eval()
    for ids, ref_seqs, alt_seqs in loader:
        ref_seqs, alt_seqs = ref_seqs.cuda(), alt_seqs.cuda()
        ref_prob = model(ref_seqs).data.cpu().numpy()
        alt_prob = model(alt_seqs).data.cpu().numpy()
        for i, seq_id in enumerate(ids):
            result[seq_id] = {'ref_prob' : ref_prob[i], 'alt_prob': alt_prob[i],
                              'abs_diff' : np.abs(ref_prob[i] - alt_prob[i])}
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inferring genomic variant effect on epigenomic profile')
    parser.add_argument('--Model', help='Model to be used for inference', type=str)
    parser.add_argument('--Device', help='CUDA device for inference', type = str, default=0)
    parser.add_argument('--BatchSize', help='Batch size for inference', type = str, default=256)
    parser.add_argument('--InputFile', help='Input seq file for inference', type=str)
    parser.add_argument('--OutDir', help='Output Directory to store result', type=str)
    parser.add_argument('--NumTarget', help='Number of epigenomic features', type = int)
    args = parser.parse_args()

    if not os.path.isfile(args.InputFile):
        print('Error: Input file not found')
        exit(1)

    if not os.path.isfile(args.Model):
        print('Error: Model not found')
        exit(1)

    if args.OutDir == None:
        print('Error: Please specify the output directory')
    elif os.path.isdir(os.path.join(args.OutDir, '')):
        pass
    else:
        cmd = 'mkdir -p ' + os.path.join(args.OurDir, '')
        os.system(cmd)

    Dset = data_loader.SNP_data(seq_path=args.InputFile)
    seq_loader = D.DataLoader(Dset, batch_size = args.BatchSize, num_workers = 0)

    device = torch.device('cpu')
    check_point = torch.load(args.Model, map_location=device)
    model = MetaChrom.MetaChrom(num_target=args.NumTarget)
    model.load_state_dict(check_point['state_dict'])

    print('------------Starting MetaChrom Inferece for Variant effect------------')
    print('Model: ' + args.Model)
    print('Number of Sequence to infer: ' + str(Dset.len))
    print('Device: ' + str(args.Device))
    print('Batch Size: ' + str(args.BatchSize))
    print('Result Directory: ' + args.OutDir)

    print('------------Running Inference------------')
    result = var_inference(model=model, loader=seq_loader, device=args.Device)

    print('------------Saving Result------------')
    result_path = os.path.join(args.OutDir, 'results.pt')
    print('Result saved at: ' + result_path)
    torch.save(result, result_path)
