# [MetaChrom]()
Ben Lai, Sheng Qian, Xin He, Jinbo Xu
[[paper]]()
[[bib]]()

**MetaChrom is a transfer learning framework that takes advantage of both an extensive compendium of publicly available chromatin profiles data, and epigenomic profiles of cell types related to specific phenotypes of interest. It's capable of predicting the genomic variant effect on epigenomic profiles with single-nucleotide resolution. Please see paper for details.**

![Image of MetaChrom](https://github.com/bl-2633/MetaChrom/blob/master/figures/MetaChrom.jpg)
*(A)Overall architecture of MetaChrom. The input sequence is fed into both the meta-feature extractor and the ResNet sequence encoder. Their outputs are then concatenated for the prediction of epigenomic profiles.(B)Pipeline for predicting variant effect on sequence epignomic profile.*

## Requirment
Training/Testing models: PyTorch >= 1.4  
Data preperation from BED and VCF files: Bedtools, twoBitToFa, Biopython

## Data preperation
Pre-trained models and processed data for demo can be downloaded [[here]]()  

To preperly run the demo notebook, use the following directory structure  
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

### Preparing variant data from rsid or vcf files

## Training

## Testing

## Citation

## License
GNU GPLv3
