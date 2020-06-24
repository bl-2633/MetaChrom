"""
Tool that converts a list of bed file to sequences and corresponding label that can be used by MetaChrom
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import os
import glob
import json
import numpy as np
import torch
import argparse
from Bio import SeqIO

def scan_input_dir(bed_dir_path):
    '''Scan and calculate metadata from the input bed directory'''
    feat_map = dict()
    bed_paths = glob.glob(bed_dir_path + '*.bed')
    for i, path in enumerate(bed_paths):
        feat_map[path.split('/')[-1].split('.')[0]] = i
    return bed_paths, feat_map

def compute_overlaps(bedtool_path, bed_path, genome_path, outdir):
    '''Compute sequence overlaps between the reference genome and input bed with bedtools2'''
    type_list = [path.split('/')[-1].split('.')[0] for path in bed_paths]
    overlap_out = outdir + 'overlap.bed'
    cmd = bedtool_path + ' -a ' + genome_path + ' -b ' + ' '.join(bed_paths) + ' -names ' + \
          ' '.join(type_list) + ' -wa -wb -f 0.5 > ' + overlap_out
    print(cmd + '\n')
    os.system(cmd)
    return

def compute_labels(OutDir, overlap_path, feat_map):
    overlap_file = open(overlap_path, 'r')
    label_save = os.path.join(OutDir, 'labels.pt')
    train_bed = open(os.path.join(OutDir, 'train.bed'), 'w')
    test_bed = open(os.path.join(OutDir, 'test.bed'), 'w')
    window_size = 1000
    genome_dict = dict()
    label_dict = dict()
    cell_type_map = feat_map
    
    for line in overlap_file:
        info = line.split('\t')
        g_id = info[3]
        label = cell_type_map[info[4]]
        if g_id not in genome_dict:
            genome_dict[g_id] = (info[0], info[1], info[2])

        if g_id not in label_dict:
            label_dict[g_id] = np.zeros(len(cell_type_map), dtype = np.uint8)
            label_dict[g_id][label] = 1
        else:
            label_dict[g_id][label] = 1
    for g_id in genome_dict:
        info = genome_dict[g_id]
        begin = int(int(info[1]) - (window_size - 200)/2)
        end = int(int(info[2]) + (window_size - 200)/2)
        
        if info[0] == 'chr7' or info[0] == 'chr8':
            test_bed.write(info[0] + '\t' + str(begin) + '\t' + str(end) + '\t' + g_id + '\n')
        else:
            train_bed.write(info[0] + '\t' + str(begin) + '\t' + str(end) + '\t' + g_id + '\n')

    train_bed.close()
    test_bed.close()
    torch.save(label_dict, label_save)
    return

def bed2fasta(OutDir, tools_path):
    twobit_path = os.path.join(tools_path, 'twoBitToFa')
    ref_path = os.path.join(tools_path, 'hg38.2bit')
    prefix = twobit_path + ' ' + ref_path + ' -noMask -bed='
    
    cmd_train = prefix + os.path.join(OutDir, 'train.bed') + ' ' + os.path.join(OutDir, 'train.fasta')
    os.system(cmd_train)
    cmd_test = cmd_train = prefix + os.path.join(OutDir, 'test.bed') + ' ' + os.path.join(OutDir, 'test.fasta')
    os.system(cmd_test)
    return

def fasta2seq(OutDir):
    train_fasta = os.path.join(OutDir, 'train.fasta')
    test_fasta = os.path.join(OutDir, 'test.fasta')
    train_seq = open(os.path.join(OutDir, 'train.seq'), 'w')
    test_seq = open(os.path.join(OutDir, 'test.seq'), 'w')
    num_train = 0
    num_test = 0

    for record in SeqIO.parse(train_fasta, 'fasta'):
        num_train += 1
        id = record.id
        seq = str(record.seq)
        train_seq.write(id + '\t' + seq + '\n')

    for record in SeqIO.parse(test_fasta, 'fasta'):
        num_test += 1 
        id = record.id
        seq = str(record.seq)
        test_seq.write(id + '\t' + seq + '\n')
    
    return num_train, num_test

if __name__ == '__main__':
    print('------------Starting Bed2Seq------------' + '\n')
    parser = argparse.ArgumentParser()
    parser.add_argument("--BedDir", help="Directory that contains bed file for processing", type=str)
    parser.add_argument("--OutDir", help="Directory that the resulting files will be stored", type=str)
    parser.add_argument("--RefGenome", help="Path to the binned reference genome", type=str, default='../../data/genome_bins.bed')
    parser.add_argument("--ToolDir", help="Directory that contains bedtools and the reference genome", default='../../tools/')
    args = parser.parse_args()
    if args.BedDir == None or not os.path.isdir(os.path.join(args.BedDir,'')):
        print('Error: Please provide the correct bed directory')
        exit(1)
    
    if args.OutDir == None:
        print('Error: Please Provide the output directory')
        exit(1)
    elif not os.path.isdir(os.path.join(args.OutDir,'')):
        os.system('mkdir ' + os.path.join(args.OutDir,''))
    else:
        pass
    MetaFile = open(os.path.join(args.OutDir, 'MetaData.txt'), 'w')
    
    print('------------Checking input BED directory------------')
    print('Input bed directory: ' + args.BedDir + '\n')
    bed_paths, feat_map = scan_input_dir(os.path.join(args.BedDir, ''))
    print('Number of Bed file Found: ' + str(len(bed_paths)))
    print(' '.join(feat_map.keys()) + '\n')
    print('Finished')

    print('------------Checking Reference Genome------------')
    if not os.path.isfile(args.RefGenome):
        print("Error: Please procide the reference genome")
        exit(1)
    print('Finished')

    print('------------Checking required tools and files------------')
    if not os.path.isfile(os.path.join(args.ToolDir, 'bedtools2/bin/intersectBed')):
        print("Error: bedtool not found")
        exit(1)
    if not os.path.isfile(os.path.join(args.ToolDir, 'hg38.2bit')):
        print("Error: 2bit referece file not found")
        exit(1)
    if not os.path.isfile(os.path.join(args.ToolDir, 'twoBitToFa')):
        print("Error: twoBitToFa not found")
        exit(1)
    print('Finished')

    print('------------Calculating overlaps------------')
    compute_overlaps(bedtool_path=os.path.join(args.ToolDir, 'bedtools2/bin/intersectBed'), 
                     bed_path=bed_paths, genome_path=args.RefGenome, outdir=args.OutDir)
    print("Writing overlap.bed")
    print("Finished")

    print("------------Composing sequence and corresponding labels------------")
    compute_labels(OutDir=os.path.join(args.OutDir, ''), overlap_path=os.path.join(args.OutDir, 'overlap.bed'), feat_map = feat_map)
    print('Writing train.bed')
    print('Writing test.bed')
    print('Writing labels.pt')
    print('Finished')

    print("------------Writing Sequence file to destination------------")
    bed2fasta(OutDir=os.path.join(args.OutDir, ''), tools_path=os.path.join(args.ToolDir, ''))
    print('Writing train.fasta')
    print('Writing test.fasta')
    num_train, num_test = fasta2seq(OutDir=os.path.join(args.OutDir, ''))
    print('Writing train.seq')
    print('Writing test.seq')
    print('Finished')

    print("------------Writing Meta data------------")
    MetaFile.write('Input bed directory: ' + args.BedDir + '\n')
    MetaFile.write('Number of BED files: ' + str(len(bed_paths)) + '\n')
    MetaFile.write('BED file names: ' + ' '.join(feat_map.keys()) + '\n')
    MetaFile.write('Number of train sequences: ' + str(num_train)+ '\n')
    MetaFile.write('Number of test sequences: ' + str(num_test) + '\n')
    torch.save(feat_map, os.path.join(args.OutDir, 'FeatMap.pt'))
    print('Finished')

    print("------------Clean Up------------")
    cmd = 'rm ' + os.path.join(args.OutDir, '*.bed')
    os.system(cmd)
    cmd = 'rm ' + os.path.join(args.OutDir, '*.fasta')
    os.system(cmd)
    print('Finished')


