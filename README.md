# Deep Learning-Based Prediction of T-Cell Receptor-Antigen Binding Recognition
Adaptive immunity is a targeted immune response that enables the body to identify and eliminate foreign pathogens, playing a critical role in the anti-tumor immune response. Tumor cell expression of antigens forms the foundation for inducing this adaptive response. However, the human leukocyte antigens (HLA)-restricted recognition of antigens by T-cell receptors (TCR) limits their ability to detect all neoantigens, with only a small subset capable of activating T-cells. Accurately predicting neoantigen binding to TCR is, therefore, crucial for assessing their immunogenic potential in clinical settings.

We present THLANet, a deep learning model designed to predict the binding specificity of TCR to neoantigens presented by class I HLAs. THLAnet employs evolutionary scale modeling-2 (ESM-2), replacing the traditional embedding methods to enhance sequence feature representation. Using scTCR-seq data, we obtained the TCR immune repertoire and constructed a TCR-pHLA binding database to validate THLANet’s clinical potential. The model’s performance was further evaluated using clinical cancer data across various cancer types. Additionally, by analyzing divided complementarity-determining region (CDR3) sequences and simulating alanine scanning of antigen sequences, we unveiled the 3D binding conformations of TCRs and antigens. Predicting TCR-neoantigen pairing remains a significant challenge in immunology; however, THLANet provides accurate predictions using only the TCR sequence (CDR3β), antigen sequence, and class I HLA, offering novel insights into TCR- antigen interactions.

# The environment of THLANet
```
python==3.9.13
numpy==1.21.2
pandas==1.4.4
torch==1.12.1
scikit-learn>=1.0.2
pandas>=1.2.4
rdkit~=2021.03.2
```

# Installation Guide
Clone this Github repo and set up a new conda environment. It normally takes about 10 minutes to install on a normal desktop computer.
```
# create a new conda environment
$ conda create --name THLAnet python=3.9.13
$ conda activate THLAnet

# install requried python dependencies
$ conda install pytorch==1.12.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
$ conda install -c conda-forge rdkit==2021.03.2
$ pip install -U scikit-learn

# clone the source code of THLAnet
$ git clone https://github.com/
$ cd THLAnet
```

# Dataset description
In this paper, epitope presentation and immunogenicity data sources are used, which are freely downloaded from IEDB (https://www.iedb.org) , 10X Genomics Datasets (https://www.10xgenomics.com/datasets) , VDJdb (https://vdjdb.cdr3.net/search).

By default, you can run our model using the immunogenicity dataset with:
```
python ESM-2-train.py

python TransformerEncoderLayer.py

python BilinearMultiHeadSelfAttention.py

python TextCNN_ESM2_train2.py

```


# Acknowledgments
The authors sincerely hope to receive any suggestions from you!

IF you have any problem, please contact us.  Email: 23B903048@stu.hit.edu.cn
## Architecture
<p align="center">
<img src="https://github.com/ChanganMakeYi/THLAnet/THLAnet.png" align="middle" height="80%" width="80%" />
</p >
