# MetaChrom
Ben Lai, Sheng Qian, Xin He, Jinbo Xu
[[paper]]()
[[bib]]()

**MetaChrom is a transfer learning framework that takes advantage of both an extensive compendium of publicly available chromatin profiles data, and epigenomic profiles of cell types related to specific phenotypes of interest. It's capable of predicting the genomic variant effect on epigenomic profiles with single-nucleotide resolution. Please see paper for details.**

![Image of MetaChrom](https://github.com/bl-2633/MetaChrom/blob/master/figures/MetaChrom.jpg)
*(A)Overall architecture of MetaChrom. The input sequence is fed into both the meta-feature extractor and the ResNet sequence encoder. Their outputs are then concatenated for the prediction of epigenomic profiles. (B)Pipeline for predicting variant effects on sequence epigenomic profile.*

## Requirment
Training/Testing models: PyTorch >= 1.4  

For data preparation, please refer to the corresponding sections below.
## Data preparation
Pre-trained models and processed data for the demo can be downloaded [[here]](blai.ttic.edu)  

To properly run the demo notebook, use the following directory structure.  
```
MetaChrom/
│   ├── tool/
│   │     ├── bedtool2/
│   │     ├── genome_bins.bed
│   │     ├── twoBitToFa
│   │     ├── hg38.2bit
│   ├── data/
│   │     ├── bed_files/
│   │     │     ├── *.bed
│   │     ├── seq_data/
│   │     │     ├── train.seq
│   │     │     ├── test.seq
│   │     │     └── labels.pt
│   │     ├── SNP_file/
│   │     │     ├── rsid.txt
│   │     │     ├── test.vcf
│   ├── trained_model/
│   │     ├── MetaChrom_models/
│   │     │     ├── MetaFeat_ResNet
│   │     ├── MetaFeat_model/
│   │     │     ├── MetaFeat
```
### Preparing data from a set of BED files  
**Requirements:**    
*Bedtools* (https://bedtools.readthedocs.io/en/latest/)  
*twoBitToFa* (http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)  
*Biopython* (https://biopython.org/)  
*The 2bit genome file corresponding to the coordinate of the bed files* (https://hgdownload.soe.ucsc.edu/downloads.html)  

### Bed2Seq.py  
Process a set of epigenomic files in BED format into sequence and feature labels  

USAGE:  
```
python3 Bed2Seq.py --BedDir <BedDir> --OutDir <OutDir> --RefGenome <binned reference genome>
                   --ToolDir <Tool directory>
```
\*\*\*\* Arguments \*\*\*\*  
```
--BedDir    : Directory that contains the list of bed files to be processed  
--OutDir    : Output directory, the program will generate labels as labels.pt and train/test sequence as train.seq and test.seq  
--RefGenome : Binned reference genome in BED format  
--ToolDir   : Directory that contains bedtools, twoBitToFa, and the 2bit genome file. 
```

\*\*\*\* Output \*\*\*\*  
Five files will be produced by this program.  
```
labels.pt   : Serialized dictionary contains the label of each sequence in train.seq and test.seq  
train.seq   : A TSV file contains training sequences with 2 fields(id, seq) each id is mapped to a key in labels.pt 
test.seq    : A TSV file contains test sequences with 2 fields(id, seq); each id is mapped to a key in labels.pt  
FeatMap.pt  : Serialized dictionary contains the mapping if label index to the corresponding BED file
MetaData.txt: Meta information of the generated data 
```
The default split of train/test sequences are based on chromosome location; all sequences located at chromosome 7 and 8 are in the test set.  

### Preparing variant data from rsid or vcf files  
**Requirment:**    
*twoBitToFa* (http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)  
*Biopython* (https://biopython.org/)  
*myvariant* (https://myvariant-py.readthedocs.io/en/latest/)  
*The 2bit genome file corresponding to the coordinate of the bed files* (https://hgdownload.soe.ucsc.edu/downloads.html)  

### SNP2Seq.py  
Process a set of variants in vcf or rsid format  

USAGE：
```
python3 SNP2Seq.py --InputType [VCF, rsid]  --InputFile <input file> --OutDir <OutDir>
                   --ToolDir <Tool directory>
```
\*\*\*\* Arguments \*\*\*\*  
```
--InputType : Type of SNP input, currently support SNV in vcf or rsid  
--InputFile : Path to the input SNP file as VCF or a list of rsid  
--OutDir    : Output directory, the program will generate labels as labels.pt and train/test sequence as train.seq and test.seq  
--ToolDir   : Directory that contains bedtools, twoBitToFa, and the 2bit genome file.   
```
\*\*\*\* Output \*\*\*\*  
One vseq files will be produced by this program.  
```
out.vseq : A TSV file contains the sequences correspond to the input SNPs with 3 fields (id, ref_seq, alt_seq)
```

## Testing/Inference

We provide two inference scripts infer.py and infer_var.py for inferring sequence epigenomic profile and genomic variant effects. 

### infer.py
Inference script for sequence epigenomic profile 

USAGE:
```
python3 infer.py --Model <trained MetaChrom model>  --InputFile <input sequence file> --OutDir <Output directory>
                 --NumTarget <number of targets of the MetaChrom model>
```
\*\*\*\* Arguments \*\*\*\*  
```
--Model     : Path to the trained MetaChrom model for inference  
--InputFile : Path to the .seq file contains sequences for inference    
--OutDir    : Output directory, where the result will be stored  
--NumTarget : Number of targets of --model 
[optional arguments]
--Device    : CUDA device for inference. default:0
--BatchSize : Size of minibatch for --model. default:256
```
\*\*\*\* Output \*\*\*\*   
```
results.pt : A serialized dictionary contains the results
{ id: predicted_epigenomic_profile}
```
### infer_var.py
Inference script for variant effects 

USAGE:
```
python3 infer_var.py --Model <trained MetaChrom model>  --InputFile <input sequence file> --OutDir <Output directory>
                 --NumTarget <number of targets of the MetaChrom model>
```
\*\*\*\* Arguments \*\*\*\*  
```
--Model     : Path to the trained MetaChrom model for inference  
--InputFile : Path to the SNP .vseq file generated by SNP2Seq.py     
--OutDir    : Output directory, where the result will be stored  
--NumTarget : Number of targets of --model 
[optional arguments]
--Device    : CUDA device for inference. default:0
--BatchSize : Size of minibatch for --model. default:256
```
\*\*\*\* Output \*\*\*\*  
```
results.pt : A serialized dictionary contains the results
{ id: 
    { 
      ref_prob: ref profile, 
      alt_prob: alt profile, 
      abs_diff: variant effect measured with absolute difference 
     } 
}
```

## Training
To train your own MetaChrom model, first prepare the training data using Bed2Seq.py as described above, then train your model with train.py

### train.py
Script for training a custom MetaChrom model  

USAGE:
```
python3 train.py --DataDir <data directory>  --ModelOut <model output directory> --BaseModel <pre-trained MetaFeat model>
                 --NumTarget <number of targets of the MetaChrom model>
```
\*\*\*\* Arguments \*\*\*\*  
```
--DataDir    : Directory that contains train.seq and labels.pt generated by Bed2Seq.py  
--ModelOut   : Directory where the trained MetaChrom model will be saved    
--BaseModel  : Path to the pre-trained MetaFeat model 
--NumTarget  : Number of targets of --model 
[optional arguments]
--Device     : CUDA device for training. default:0
--BatchSize  : Size of minibatch for --model. default:256
--lr         : Learning rate for the Adam optimizer. default:1e-3
--Epoch      : Number of Epoch for training. default:50
```

\*\*\*\* Output \*\*\*\*  
```
MetaFeat_ResNet : Trained MetaChrom model stoed at <OutDir>
```


## Citation

## License
GNU GPLv3
