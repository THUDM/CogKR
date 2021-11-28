# CogKR
[CogKR: Cognitive Graph for Multi-hop Knowledge Reasoning](https://www.computer.org/csdl/journal/tk/5555/01/09512424/1w0wzCuvnA4)

Accepted to IEEE TKDE.

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
Then install [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter) manually.

### Dataset
Three public datasets FB15K-237, WN18RR, and YAGO3-10 are used for knowledge graph completion. The original datasets can be downloaded from [FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312), [WN18RR](https://github.com/TimDettmers/ConvE), and [YAGO3-10](https://github.com/TimDettmers/ConvE/raw/master/YAGO3-10.tar.gz).

Two public datasets NELL-One and Wiki-One (slightly modified) are used for one-shot link prediction. The original datasets can be downloaded from [One-shot Relational Learning](https://github.com/xwhan/One-shot-Relational-Learning). You can download the preprocessed datasets from the [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/El-XlZVxAtNMkVTUN5-KB5gBupAOgY-qMVvf702aVceIgw?e=LcWwqz) in OneDrive. If you're in regions where OneDrive is not available (e.g. Mainland China), try to the [link](https://cloud.tsinghua.edu.cn/d/4ba979c61b6f40cc9be8/) in Tsinghua Cloud.

After downloading the dataset, please unzip it into the `datasets/{dataset_name}/data` folder.

To use your own dataset, see the "Use your dataset" part below.

### Preprocess

```shell
python src/preprocess.py --directory datasets/{dataset_name} --process_data --save_train
```

### Training

For training, simply sun

```shell
python src/main.py --directory datasets/{dataset_name} --gpu {gpu_id} --config {config_file} --comment {experiment_name}
```

Use `dataset_path` to specify the path to the dataset.

Use `gpu_id` to specify the id of the gpu to use.

`config_file` is used to specify the configuration file for experimental settings and  hyperparameters. Different configurations for two datasets in the paper are stored under the `configs/` folder.

`experiment_name` is used to specify the name of the experiment.

### Evaluation

For evaluation, simply run

```shell
python src/main.py --inference --directory {dataset_path} --gpu {gpu_id} --config {config_file} --load_state {state_file}
```

### Use Your Own Dataset

To use your own dataset, please put the files of the dataset under `datasets/` in the following structure:

```
-{dataset_name}/data
    -train.txt
    -valid_support.txt
    -valid_eval.txt
    -test_support.txt
    -test_eval.txt
    -ent2id.txt (optional)
    -relation2id.txt (optional)
    -entity2vec.{embed_name} (optional)
    -relation2vec.{embed_name} (optional)
    -rel2candidates.json (optional)
```

`train.txt`,`valid_support.txt`, `valid_eval.txt`, `test_support.txt` and `test_eval.txt` correspond to the facts of training relations, support facts and evaluate facts of validation relations and support facts and evaluate facts of test relations, for one-shot link prediction tasks. Each line is in the format of `{head}\t{relation}\t{tail}\n`. For knowledge graph completion, `train.txt`, `valid_eval.txt`, and `test_eval.txt` should be the train, valid, and test sets. `valid_support.txt` and `test_support.txt` should be empty.

`ent2id.txt`, `relation2id.txt`, `entity2vec.{embed_name}` and  `relation2vec.{embed_name}` are used for pretrained KG embeddings. The usage of pretrained embeddings is not required. Each line of `ent2id.txt` or `relation2id.txt` is the entity/relation name whose id is the line number(starting from 0). Each line of `entity2vec.{embed_name}` or `relation2vec.{embed_name}` is the vector of the entity/relation whose id is the line number.

`rel2candidates.json` represents the candidate entities of test and validation relations. The file is only used for one-shot link prediction in our experiment.

## Cite

Please cite our paper if you use the code or datasets in your own work:

```
@ARTICLE {9512424,
author = {Z. Du and C. Zhou and J. Yao and T. Tu and L. Cheng and H. Yang and J. Zhou and J. Tang},
journal = {IEEE Transactions on Knowledge & Data Engineering},
title = {CogKR: Cognitive Graph for Multi-hop Knowledge Reasoning},
year = {5555},
volume = {},
number = {01},
issn = {1558-2191},
pages = {1-1},
keywords = {cognition;task analysis;urban areas;training;computational modeling;benchmark testing;scalability},
doi = {10.1109/TKDE.2021.3104310},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {aug}
}
```
