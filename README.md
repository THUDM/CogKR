# CogKR

### [arXiv](https://arxiv.org/abs/1906.05489)

Cognitive Knowledge Graph Reasoning for One-shot Relational Learning

Zhengxiao Du, Chang Zhou, Ming Ding, Hongxia Yang, Jie Tang

Under Review in NeurIPS 2019

**Under construction**.

## Prerequisites

* Python 3
* PyTorch >= 1.1.0
* NVIDIA GPU + CUDA cuDNN

## Getting Started

### Installation

Clone this repo

```shell
git clone https://github.com/THUDM/CogKR
cd CogKR
```

Please install dependencies by

```shell
pip install -r requirements.txt
```

### Dataset

Two public datasets NELL-One and Wiki-One (slightly modified) are used in our experiment. The original datasets can be downloaded from [here](https://github.com/xwhan/One-shot-Relational-Learning).

You can download the preprocessed datasets from the [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/El-XlZVxAtNMkVTUN5-KB5gBupAOgY-qMVvf702aVceIgw?e=LcWwqz) in OneDrive. If you're in regions where OneDrive is not available (e.g. Mainland China), try to the [link](https://cloud.tsinghua.edu.cn/d/4ba979c61b6f40cc9be8/) in Tsinghua Cloud.

After downloading the dataset, please unzip it into the datasets folder.

To use your own dataset, see the "Use your dataset" part below.

### Training

For training, simply sun

```shell
python src/main.py --directory {dataset_path} --gpu {gpu_id} --config {config_file} --load_embed DistMult --comment {experiment_name}
```

Use `dataset_path` to specify the path to the dataset, which in our expeirment should be `datasets/NELL` or `datasets/Wiki`.

Use `gpu_id` to specify the id of the gpu to use.

`config_file` is used to specify the configuration file for experimental settings and  hyperparameters. Different configurations for two datasets in the paper are stored under the `configs/` folder. `config-nell.json` and `config-wiki.json` are used to train the complete model. `config-nell-onlyr.json` and `config-wiki-onlyr.json` are used to train the CogKR-onlgR model for abalation study.

`experiment_name` is used to specify the name of the experiment.

If you suffer from **out of memory** error when running experiments on Wiki-One, try to run the code with `--sparse_embed` to use sparse gradient for the embedding layer

### Use Your Own Dataset

To use your own dataset, please put the files of the dataset under `datasets/` in the following structure:

```
-{dataset_name}
	-data
    -train.txt
    -valid_support.txt
    -valid_eval.txt
    -test_support.txt
    -test_eval.txt
    -ent2id.txt (optional)
    -relation2id.txt (optional)
    -entity2vec.{embed_name}
    -relation2vec.{embed_name}
    -rel2candidates.json (optional)
```

`train.txt`,`valid_support.txt`, `valid_eval.txt`, `test_support.txt` and `test_eval.txt` correspond to the facts of training relations, support facts and evaluate facts of validation relations and support facts and evaluate facts of test relations. Each line is in the format of `{head}\t{relation}\t{tail}\n`.

`ent2id.txt`, `relation2id.txt`, `entity2vec.{embed_name}` and  `relation2vec.{embed_name}` are used for pretrained KG embeddings. The usage of pretrained embeddings is not required but highly recommended. Each line of `ent2id.txt` or `relation2id.txt` is the entity/relation name whose id is the line number(starting from 0). Each line of `entity2vec.{embed_name}` or `relation2vec.{embed_name}` is the vector of the entity/relation whose id is the line number.

`rel2candidates.json` represents the candidate entities of test and validation relations.

Firstly, preprocess the data

```shell
python src/main.py --directory datasets/{dataset_name} --process_data
```

Then you can train the model according to the "Training" part.

There are also two files  `evaluate_graphs` and `fact_dist` in our preprocessed dataset. `fact_dist` is used to skip some facts in the training set. To generate the file, please run

```shell
python src/main.py --directory datasets/{dataset_name} --config {config_file} --get_fact_dist
```

To generate `evaluate_graphs`, please run

```shell
python src/main.py --directory datasets/{dataset} --search_evaluate_graph
```

Or on the Wiki-One dataset only:

```shell
python src/main.py --directory datasets/Wiki --search_evaluate_graph --wiki
```

## Cite

Please cite our paper if you use the code or datasets in your own work:
```
@article{du2019cogkr,
  author    = {Zhengxiao Du and
               Chang Zhou and
               Ming Ding and
               Hongxia Yang and
               Jie Tang},
  title     = {Cognitive Knowledge Graph Reasoning for One-shot Relational Learning},
  journal   = {CoRR},
  volume    = {abs/1906.05489},
  year      = {2019}
}
```
