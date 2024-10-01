# Scrambledtext Analysis

Can synthetic corrupted data be used to train LMs to correct OCR text?

This repo contains the code used to prepare and analyse the paper "[Scrambled text: training Language Models to correct OCR errors using synthetic data](https://arxiv.org/abs/2409.19735)". The paper abstract is below.

> OCR errors are common in digitised historical archives, significantly affecting their usability and value.
> Generative Language Models (LMs) have shown potential for correcting these errors using the context provided by the corrupted text and the broader socio-cultural context, a process called Context Leveraging OCR Correction (CLOCR-C). However, getting sufficient training data for fine-tuning such models can prove challenging. This paper shows that fine-tuning a language model on synthetic data using an LM and using a character level Markov corruption process can significantly improve the ability to correct OCR errors. Models trained on synthetic data reduce the character error rate by 55% and word error rate by 32% over the base LM and outperform models trained on real data. Key findings include; training on under-corrupted data is better than over-corrupted data; non-uniform character level corruption is better than uniform corruption; More tokens-per-observation outperforms more observations for a fixed token budget. The outputs for this paper are a set of 8 heuristics for training effective CLOCR-C models, a dataset of 10,000 synthetic 19th century newspaper articles and \verb|scrambledtext| a python library for creating synthetic corrupted data.


# Related repos


This repo is part of the larger ScrambledText project:

- [training_lms_with_synthetic_data](https://github.com/JonnoB/training_lms_with_synthetic_data): Contains code for training the Language Models.
- [scrambledtext](https://github.com/JonnoB/scrambledtext): Library used to create the synthetic data.

# Notebooks

To help with reproducibility, this repo uses [marimo](https://github.com/marimo-team/marimo). Marimo notebooks are stored with

- `create_synthetic_dataset.py`: The code used to create the synthetic 19th-century newspaper articles.
- `corrupt_synthetic_dataset.py`: Creating various examples of corrupted data. However, almost all the training data is corrupted during the training process.
- `corruption2.py`: Additional corruption experiments, mainly used for the supplementary materials.
- `analysing_models.py`: Code used to analyse the output of the models trained in the  [training_lms_with_synthetic_data](https://github.com/JonnoB/training_lms_with_synthetic_data) repo.

# Auxiliary scripts

The below scripts are used to store the functions used in the project.

- `scrambledtext.py`: The version of scrambled text used in the project.
- `lm_support_functions.py`: Functions used to help generate synthetic news articles using GPT.


# Dependencies
Key dependencies are

- `marimo`: Notebook system used by the project
- `uv`: Virtual environment and package dependency manager. It isn't essential but is recommended as it makes life much easier.

For a full list see `requirements.txt`

# Getting started

Clone the repository

```
git clone https://github.com/your-username/scrambledtext-analysis.git
cd scrambled text-analysis
```

Set up a virtual environment (using uv)

```
uv venv
source .venv/bin/activate
```

Install dependencies

```
uv pip install -r requirements.txt
```

Run a notebook

```
marimo edit create_synthetic_dataset.py
```

# Data

The key datasets necessary for running this repo are available from the UCL data repository [Scrambled text Datasets from the paper](https://doi.org/10.5522/04/27108334.v1). This data should be placed inside this repo's `data` folder.


# License

This project is licensed under the MIT License - see the LICENSE.md file for details.

# Citing this repo

If this repo is helpful in your work, please cite the Arxiv pre-print

Scrambled text: training Language Models to correct OCR errors using synthetic data: https://arxiv.org/abs/2409.19735