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
Pre-trained models and processed data for the demo can be downloaded [[here]]()  

To properly run the demo notebook, use the following directory structure.  
```
MetaChrom/
│   ├── data/
│   │     ├── seq_data/
│   │     │     ├── train.seq
│   │     │     ├── test.seq
│   │     │     └── labels.pt
│   │     ├── SNP_data/
│   │     │     ├──test_SNP.seq
│   ├── trained_model/
│   │     ├── MetaChrom_models/
│   │     │     ├── MetaFeat_ResNet
│   │     ├── MetaFeat_model/
│   │     │     ├── MetaFeat
```
### Preparing data from a set of BED files  
Requirment:  
Bedtools(https://bedtools.readthedocs.io/en/latest/)  
twoBitToFa(http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)  
Biopython(https://biopython.org/)  
The 2bit genome file corresponding to the coordinate of the bed files(https://hgdownload.soe.ucsc.edu/downloads.html)  

Bed2Seq  
Process a set of epigenomic files in BED format into sequence and feature labels  

USAGE:  
```
python3 Bed2Seq.py --BedDir <BedDir> --OutDir <OutDir> --RefGenome <binned reference genome>
                   --ToolDir <Tool directory>
```
\*\*\*\* Required argument \*\*\*\*  
--BedDir    : Directory that contains the list of bed files to be processed  
--OutDir    : Output directory, the program will generate labels as labels.pt and train/test sequence as train.seq and test.seq  
--RefGenome : Binned reference genome in BED format  
--ToolDir   : Directory that contains bedtools, twoBitToFa, and the 2bit genome file. 

\*\*\*\* Output \*\*\*\*  
Two files will be produced by this program. labels.pt is a serialized dictionary contains label for each sequence in train.seq and test.seq  
.seq file is a TSV with 2 fields(id, seq) each id is mapped to a key in labels.pt readable by data_loader  
### Preparing variant data from rsid or vcf files  
Requirment:  
twoBitToFa(http://hgdownload.soe.ucsc.edu/downloads.html#source_downloads)  
Biopython(https://biopython.org/)  
myvariant(https://myvariant-py.readthedocs.io/en/latest/)  
The 2bit genome file corresponding to the coordinate of the bed files(https://hgdownload.soe.ucsc.edu/downloads.html)  

SNP2Seq  
Process a set of variants in vcf or rsid format  

USAGE：
```
python3 SNP2Seq.py --InputType [VCF, rsid]  --InputFile <input file> --OutDir <OutDir>
                   --ToolDir <Tool directory>
```
\*\*\*\* Required argument \*\*\*\*  
--InputType : Type of SNP input, curretnly support SNV in vcf or rsid  
--InputFile : Path to the input SNP file as VCF or a list of rsid  
--OutDir    : Output directory, the program will generate labels as labels.pt and train/test sequence as train.seq and test.seq  
--ToolDir   : Directory that contains bedtools, twoBitToFa, and the 2bit genome file.   
\*\*\*\* Output \*\*\*\*  
A seq file will be generated as TSV with three fields(id, ref_seq, alt_seq) readable by data_loader  
## Training

## Testing/Inference

## Citation

## License
GNU GPLv3
