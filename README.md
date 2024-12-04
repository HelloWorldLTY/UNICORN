# UNICORN

This is the official code repo of the paper: UNICORN: Towards Universal Cellular Expression Prediction with an Explainable Multi-Task Learning Framework. 

# Installation

To install UNICORN, please refer to the content in the yml file "unicorn.yml". Ideally, it is the same as required packages from [seq2cells](https://github.com/GSK-AI/seq2cells).

If you are interested in individualized gene expression prediction, you may need to install [bcftools](https://samtools.github.io/bcftools/bcftools.html) and [samtools](https://www.htslib.org/).

```
conda install bcftools
conda install samtools
```

These instructions can help you prepare all the required files for running UNICORN.

# Model

Please check the folder **seq2cells** for details of the model structure. The files under the models contain the variations used in UNICORN.

If you intend to use different DNA sequence embeddings for predicting gene expression, please refer [BEND](https://github.com/frederikkemarin/BEND) for details.

If you intend to use gene embeddings for natural language, please refer [scELMo](https://sites.google.com/yale.edu/scelmolib) for details.

# Tutorial

We provide a tutorial for training and evaluating UNICORN in the tutorial folder. 

# Datasets

Please check [this link](https://zenodo.org/records/8314644) for accessing the datasets used for training (We will update our manuscript for the full information soon).

To use GTEx data, please visit [this website](https://gtexportal.org/home/eqtlDashboardPage) to request access to protected data.

# Acknowledgements

We thank the authors of seq2cells, bcftools, and samtools for their great code design.

# Citation
