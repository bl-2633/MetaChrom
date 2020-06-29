"""
Tool that converts SNP file to sequences and corresponding label that can be used by MetaChrom
"""
__author__ = "Ben Lai"
__copyright__ = "Copyright 2020, TTIC"
__license__ = "GNU GPLv3"

import os
from Bio import SeqIO
import argparse
import myvariant

def vcf2bed(vcf_file, OutDir):
    window_size = 1000
    vcf_file = open(vcf_file, 'r')
    bed_file = open(os.path.join(OutDir, 'tmp.bed'), 'w')
    num_input = 0
    num_out = 0
    for line in vcf_file:
        num_input += 1
        info = line.strip().split('\t')
        if len(info) < 5:
            print("Error: Incorrect VCF format")
            exit(1)
        chrom = info[0]
        pos = info[1]
        id = info[2]
        ref = info[3]
        alt = info[4]
        if len(ref) != 1 or len(alt) != 1:
            print("Error: Only SNVs are supported")
            continue
        begin = int(int(pos) - window_size/2)
        end = int(int(pos) + window_size/2)
        bed_file.write(chrom + '\t' + str(begin) + '\t' + str(end) + '\t' +  id + ';' + ref + ';' + alt + '\n')
        num_out += 1
    return num_input, num_out

def rsid2bed(rsid_file, OutDir):
    window_size = 1000
    mv = myvariant.MyVariantInfo()
    rsid_file = open(rsid_file, 'r')
    bed_file = open(os.path.join(OutDir, 'tmp.bed'), 'w')
    num_input = 0
    num_out = 0
    for line in rsid_file:
        num_input += 1
        if line[:2] != 'rs':
            print("Error: Please input valid rsid")
        info = mv.query(line, assembly = 'hg38')
        if len(info['hits']) == 0:
            continue
        chrom = info['hits'][0]['chrom']
        pos = info['hits'][0]['vcf']['position']
        ref = info['hits'][0]['vcf']['ref']
        alt = info['hits'][0]['vcf']['alt']
        id  = line.strip()

        begin = int(int(pos) - window_size/2)
        end = int(int(pos) + window_size/2)
        bed_file.write('chr' + chrom + '\t' + str(begin) + '\t' + str(end) + '\t' +  id + ';' + ref + ';' + alt + '\n')
        num_out += 1
    return num_input, num_out

def bed2seq(OutDir, ToolDir):
    bed_file = os.path.join(OutDir, 'tmp.bed')
    fasta_file = os.path.join(OutDir, 'tmp.fasta')
    seq_file = open(os.path.join(OutDir, 'out.vseq'), 'w')
    twobit_path = os.path.join(ToolDir, 'twoBitToFa')
    ref_path = os.path.join(ToolDir, 'hg38.2bit')
    window_size = 1000

    prefix = twobit_path + ' ' + ref_path + ' -noMask -bed='
    cmd = prefix + bed_file + ' ' + fasta_file
    os.system(cmd)

    for record in SeqIO.parse(fasta_file, 'fasta'):
        ref_seq = str(record.seq)
        alt_seq = list(str(record.seq))
        alt = record.id.split(';')[2]
        alt_seq[int(window_size/2 - 1)] = alt
        alt_seq = ''.join(alt_seq)
        seq_file.write(record.id + '\t' + ref_seq + '\t' + alt_seq + '\n')

    return


if __name__ == '__main__':
    print('------------Starting SNP2Seq------------' + '\n')
    parser = argparse.ArgumentParser()
    parser.add_argument("--InputType", help="Input format of SNPs. VCF or rsid", type=str)
    parser.add_argument("--InputFile", help="Path to the input file", type = str)
    parser.add_argument("--ToolDir",help="Directory that contains bedtools and the reference genome", default='../../tools/')
    parser.add_argument("--OutDir", help="Path to the output directory", type = str)
    args = parser.parse_args()

    if args.InputFile == None or not os.path.isfile(args.InputFile):
        print("Error: Input file not found")
        exit(1)

    if args.InputType not in ['rsid', 'VCF']:
        print("Error: Input format not supported")
        exit(1)

    if args.OutDir == None:
        print('Error: Please Provide the output directory')
        exit(1)
    elif not os.path.isdir(os.path.join(args.OutDir,'')):
        os.system('mkdir ' + os.path.join(args.OutDir,''))
    else:
        pass

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


    print('------------Converting input to bed------------')
    if args.InputType == 'VCF':
        vcf2bed(vcf_file=args.InputFile, OutDir=args.OutDir)
        pass
    else:
        rsid2bed(rsid_file=args.InputFile, OutDir=args.OutDir)
        pass
    print('Finished')

    print('------------Writing seq file to destination------------')
    bed2seq(OutDir=args.OutDir, ToolDir=args.ToolDir)
    print('Finished')

    print('------------Clean up------------')
    cmd = 'rm ' + os.path.join(args.OutDir, '*.bed')
    os.system(cmd)
    cmd = 'rm ' + os.path.join(args.OutDir, '*.fasta')
    os.system(cmd)
    print('Finished')



