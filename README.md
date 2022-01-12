# Annotating functional effects of non-coding variants in neuropsychiatric cell types by Deep Transfer Learning(MetaChrom)

[[Paper]](https://www.biorxiv.org/content/10.1101/2021.02.02.429064v1.abstract)  
[[WebServer]](https://metachrom.ttic.edu/)

**This is the official code and data repository for the paper "Annotating functional effects of non-coding variants in neuropsychiatric cell types by Deep Transfer Learning".**
<!---
![Image of MetaChrom](https://github.com/bl-2633/MetaChrom/blob/master/figures/MetaChrom.jpg)
*(A)Overall architecture of MetaChrom. The input sequence is fed into both the meta-feature extractor and the ResNet sequence encoder. Their outputs are then concatenated for the prediction of epigenomic profiles. (B)Pipeline for predicting variant effects on sequence epigenomic profile.*
-->
## Usage
Pre-trained models and data for the demo can be downloaded [[here]](http://blai.ttic.edu/data/metachrom/)  
For neural developmental MetaChrom models demonstraed in our paper please see the [section](#MetaChrom-trained-in-neural-development-context) below

To train and build variant interpretation pipeline with MetaChrom consists of 4 consecutive steps:  
[1. Preparing bed files of the cellular context of interests](#Prepare-BED-data-for-model-training)  
[2. Prepare variant data in vcf format and process them](#Prepare-variant-files-for-evaluating-variant-effects)  
[3. Train a MetaChrom model using the prepared data](#Training)   
[4. Compute epigenomic profiles and variant effects](#Inference)  

## Requirment
Training/Testing models: PyTorch >= 1.4  

For dataprepertaion the following external tools are needed:  
*Bedtools* (https://bedtools.readthedocs.io/en/latest/)  
*twoBitToFa* (http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)  
*Biopython* (https://biopython.org/)  
*The 2bit genome file corresponding to the coordinate of the bed files* (https://hgdownload.soe.ucsc.edu/downloads.html)  
*myvariant* (https://myvariant-py.readthedocs.io/en/latest/)   

## Prepare BED data for model training
To prepare BED files in to sequence files, use ```Bed2Seq.py``` as described below, the output files can be loaded by ```data_loader.py``` 
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

## Prepare variant files for evaluating variant effects
To prepare the variant for effect assessment, use ```SNP2Seq.py``` in vcf format or rsid list. 
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

## Training
To train your own MetaChrom model, use ```train.py``` with data prepared by ```Bed2Seq.py```

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


## Inference

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

## MetaChrom trained in neural development context
To properly run the notebooks, use the following directory structure.  
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
To train/test model deomonstrated in the paper, dolwnload the bed files and pre-trained MetaFeat model in it's corresponding directories.
### Processing neural developmental data
```python ./src/data_processing/ed2Seq.py --BedDir ./data/bed_files/ --OutDir ./data/seq_data/ --RefGenome ./tool/genome_bins.bed --ToolDir ./tool/```  
The processed data will be deposited in ```./data/seq_data/``` and it's ready to be used for training.
### Training MetaChrom
```python ./src/train.py --DataDir ./data/seq_data/ --ModelOut ./trained_model/neural_MetaChrom/ --NumTarget 31 --BaseModel ./trained_models/MetaFeat_model/MetaFeat```  
The trained model will be deposited in ```./trained_mdoel/neural_MetaChrom/```
### Demos for epigenomic profile and variant effect inference
We also provided a pre-trained neural developmental MetaChrom model for running our program locally. We also have a webserver for small batch inference at ```https://metachrom.ttic.edu/```.  There is an example [jupyter notebooks](https://github.com/bl-2633/MetaChrom/tree/master/Demo) available. 

## Citation

## License
GNU GPLv3
