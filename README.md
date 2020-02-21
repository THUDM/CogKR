# CogKR
**Under construction**.

## Results
<table>
   <tr>
      <td></td>
      <td>FB15K-237</td>
      <td></td>
      <td></td>
      <td></td>
      <td>WN18RR</td>
      <td></td>
      <td></td>
      <td></td>
   </tr>
   <tr>
      <td>Model</td>
      <td>Hits@1</td>
      <td>Hits@3</td>
      <td>Hits@10</td>
      <td>MRR</td>
      <td>Hits@1</td>
      <td>Hits@3</td>
      <td>Hits@10</td>
      <td>MRR</td>
   </tr>
   <tr>
      <td>TransE </td>
      <td> - </td>
      <td> - </td>
      <td>46.5</td>
      <td>29.4</td>
      <td> - </td>
      <td> </td>
      <td>50.1</td>
      <td>22.6</td>
   </tr>
   <tr>
      <td>DistMult </td>
      <td>20.6</td>
      <td>31.8</td>
      <td> - </td>
      <td>29</td>
      <td>38.4</td>
      <td>42.4</td>
      <td> - </td>
      <td>41.3</td>
   </tr>
   <tr>
      <td>ComplEx </td>
      <td>20.8</td>
      <td>32.6</td>
      <td> - </td>
      <td>29.6</td>
      <td>38.5</td>
      <td>43.9</td>
      <td> - </td>
      <td>42.2</td>
   </tr>
   <tr>
      <td>ConvE </td>
      <td>23.7</td>
      <td>35.6</td>
      <td>50.1</td>
      <td>32.5</td>
      <td>40</td>
      <td>44</td>
      <td>52</td>
      <td>43</td>
   </tr>
   <tr>
      <td>RotatE </td>
      <td>24.1</td>
      <td>37.5</td>
      <td>53.3</td>
      <td>33.8</td>
      <td>42.8</td>
      <td>49.2</td>
      <td><b>57.1</b></td>
      <td>47.6</td>
   </tr>
   <tr>
      <td>TuckER </td>
      <td>26.6</td>
      <td>39.4</td>
      <td>54.4</td>
      <td>35.8</td>
      <td>44.3</td>
      <td>48.2</td>
      <td>52.6</td>
      <td>47</td>
   </tr>
   <tr>
      <td>NeuralLP </td>
      <td>18.2</td>
      <td>27.2</td>
      <td> - </td>
      <td>24.9</td>
      <td>37.2</td>
      <td>43.4</td>
      <td> - </td>
      <td>43.5</td>
   </tr>
   <tr>
      <td>MINERVA </td>
      <td>21.7</td>
      <td>32.9</td>
      <td>45.6</td>
      <td>29.3</td>
      <td>41.3</td>
      <td>45.6</td>
      <td>51.3</td>
      <td>44.8</td>
   </tr>
   <tr>
      <td>M-Walk </td>
      <td>16.5</td>
      <td>24.3</td>
      <td> - </td>
      <td>23.2</td>
      <td>41.4</td>
      <td>44.5</td>
      <td> - </td>
      <td>43.7</td>
   </tr>
   <tr>
      <td>DPMPN </td>
      <td>28.6</td>
      <td>40.3</td>
      <td>53</td>
      <td>36.9</td>
      <td>44.4</td>
      <td>49.7</td>
      <td>55.8</td>
      <td>48.2</td>
   </tr>
   <tr>
      <td>CogKR </td>
      <td><b>34.5</b></td>
      <td><b>47.3</b></td>
      <td><b>59.3</b></td>
      <td><b>42.6</b></td>
      <td><b>45.9</b></td>
      <td><b>50.8</b></td>
      <td>55.7</td>
      <td><b>49.1</b></td>
   </tr>
</table>

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
Two public datasets FB15K-237 and WN18RR are used for knowledge graph completion. The original datasets can be downloaded from [FB15K-237](https://www.microsoft.com/en-us/download/details.aspx?id=52312) and [WN18RR](https://github.com/TimDettmers/ConvE).

Two public datasets NELL-One and Wiki-One (slightly modified) are used for one-shot link prediction. The original datasets can be downloaded from [One-shot Relational Learning](https://github.com/xwhan/One-shot-Relational-Learning).

You can download the preprocessed datasets from the [link](https://mailstsinghuaeducn-my.sharepoint.com/:f:/g/personal/duzx16_mails_tsinghua_edu_cn/El-XlZVxAtNMkVTUN5-KB5gBupAOgY-qMVvf702aVceIgw?e=LcWwqz) in OneDrive. If you're in regions where OneDrive is not available (e.g. Mainland China), try to the [link](https://cloud.tsinghua.edu.cn/d/4ba979c61b6f40cc9be8/) in Tsinghua Cloud.

After downloading the dataset, please unzip it into the datasets folder.

To use your own dataset, see the "Use your dataset" part below.

### Training

For training, simply sun

```shell
python src/main.py --directory {dataset_path} --gpu {gpu_id} --config {config_file} --comment {experiment_name}
```

Use `dataset_path` to specify the path to the dataset, which in our expeirment should be `datasets/NELL` or `datasets/Wiki`.

Use `gpu_id` to specify the id of the gpu to use.

`config_file` is used to specify the configuration file for experimental settings and  hyperparameters. Different configurations for two datasets in the paper are stored under the `configs/` folder. `config-nell.json` and `config-wiki.json` are used to train the complete model. `config-nell-onlyr.json` and `config-wiki-onlyr.json` are used to train the CogKR-onlgR model for abalation study.

`experiment_name` is used to specify the name of the experiment.

### Evaluation

For evaluation, simply run

```shell
python src/main.py --inference --directory {dataset_path} --gpu {gpu_id} --config {config_file} --load_state {state_file}
```

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

`train.txt`,`valid_support.txt`, `valid_eval.txt`, `test_support.txt` and `test_eval.txt` correspond to the facts of training relations, support facts and evaluate facts of validation relations and support facts and evaluate facts of test relations, for one-shot link prediction tasks. Each line is in the format of `{head}\t{relation}\t{tail}\n`. For knowledge graph completion, `train.txt`, `valid_eval.txt`, and `test_eval.txt` should be the train, valid, and test sets. `valid_support.txt` and `test_support.txt` should be empty.

`ent2id.txt`, `relation2id.txt`, `entity2vec.{embed_name}` and  `relation2vec.{embed_name}` are used for pretrained KG embeddings. The usage of pretrained embeddings is not required but highly recommended. Each line of `ent2id.txt` or `relation2id.txt` is the entity/relation name whose id is the line number(starting from 0). Each line of `entity2vec.{embed_name}` or `relation2vec.{embed_name}` is the vector of the entity/relation whose id is the line number.

`rel2candidates.json` represents the candidate entities of test and validation relations.

Firstly, preprocess the data

```shell
python src/main.py --directory datasets/{dataset_name} --process_data
```

Then you can train the model according to the "Training" part.

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
