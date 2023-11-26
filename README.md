<h1 align="center">
<img style="vertical-align:middle" width="400" height="180" src="https://raw.githubusercontent.com/wln20/Retrieval_QA/master/docs/logo.jpg" />
</h1>

<p align="center">
    <a href="https://github.com/wln20/Retrieval_QA">
        <img alt="develop" src="https://img.shields.io/badge/develop-v0.0-blue">
    </a>
    <a href="https://www.python.org/">
            <img alt="build" src="https://img.shields.io/badge/build-python-green">
    </a>
    <a href="https://github.com/wln20/Retrieval_QA/blob/master/LICENSE">
        <img alt="license" src="https://img.shields.io/badge/license-Apache_2.0-red">
    </a>
    <a href="https://github.com/wln20/Retrieval_QA/blob/master/raw_data">
        <img alt="download" src="https://img.shields.io/badge/download-raw-blue">
    </a>
      <a href="https://huggingface.co/datasets/lnwang/retrieval_qa">
        <img alt="huggingface" src="https://img.shields.io/badge/huggingface-dataset-yellow">
    </a>
    
  
</p>

### Introduction
#### About this repository
This repository contains the raw data of the dataset `Retrieval_QA`, along with example scripts to do evaluation on your customized models with `Retrieval_QA`.
#### About the dataset
##### Basic information
The purpose of `Retrieval_QA` is to provide a simple and easy-to-use multilingual benchmark for retrieval encoder models, which helps researchers quickly select the most effective retrieval encoder for text extraction and achieve optimal results in subsequent retrieval tasks such as retrieval-augmented-generation (RAG). The dataset contains multiple document-question pairs, where each document is a short text about the history, culture, or other information of a country or region, and each question is a query relevant to the content of the corresponding document.

Users may select a retrieval encoder model to encode each document and query into corresponding embeddings, and then use vector matching methods such as FAISS to identify the most relevant documents for each query as retrieval results. Then you may use the `acc ~ top-k` graph as a metric to evaluate whether this model act as a good encoder for retrieval.

+ Curated by: <a href='https://wln20.github.io'>Luning Wang</a>
+ Language(s): English, Chinese(Simplified, Traditional)
+ License: Apache-2.0

##### Data source
The raw data was generated by GPT-3.5-turbo, using carefully designed prompts by human. The data was also cleaned to remove controversial information.

Now we support English(en), Simplified Chinese(zh_cn), Traditional Chinese(zh_tw).

### Usage
#### Environment setup
- Clone this repository:
    ```bash
    git clone https://github.com/wln20/Retrieval_QA.git
    cd Retrieval_QA
    ```
- Create a new conda environment and install the dependencies:
  ```bash
  conda create -n rqa python==3.10
  conda activate rqa
  pip install -r requirements.txt
  ```
#### Example usage
The dataset is now available on 🤗 Huggingface, you can conveniently use it in python with 🤗 Datasets:
```python
from datasets import load_dataset
dataset = load_dataset('lnwang/retrieval_qa', name='en')
```
### Trouble Shooting
1. If you're using baichuan model and encounter this error: `AttributeError: 'BaichuanTokenizer' object has no attribute 'sp_model'`，the following solution may help: https://github.com/baichuan-inc/Baichuan2/issues/204#issuecomment-1756867868
