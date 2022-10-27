# Code for TopKG
> The implementation of our paper accepted by COLING2022: [TopKG: Target-oriented Dialog via Global Planning on Knowledge Graph](https://aclanthology.org/2022.coling-1.62/)

## Prerequisites

make sure python >= 3.7
Install the required packages:

```python
pip install -r requirements.txt
```



## Directory Structure

If you want to run the process step from original data, please download the relevant files and put them in `raw_files` folder

```tree
|-- cache
    |-- bert # for discriminator
    |-- dev # for generator
|-- code
|-- data # processed json data and graph
|-- output # output result for evalution
|-- preprocess # process original data
|-- raw_files
    |-- ConvAI2 # Raw data downloaded from https://parl.ai/projects/convai2/
        |-- train.txt
        |-- test.txt
    |-- OTTers # Raw data downloaded from https://github.com/karinseve/OTTers
        |-- dev
            |-- source.csv
            |-- target.txt
        |-- test
            |-- source.csv
            |-- target.txt
        |-- train
            |-- source.csv
            |-- target.txt
    |-- conceptnet-assertions-5.7.0.csv # Raw file downloaded from https://github.com/commonsense/conceptnet5
    |-- glove.6B.300d.txt
```

## Data Preparation

Execute the following commands step by step

```bash
python preprocess/extract_english.py # extract english triple from conceptnet
python preprocess/ground_concepts.py # Extract keywords/concepts/entities from the dialogue
python preprocess/otter_reason.py # Constructing dialogue map for otters corpus
python preprocess/tgconv_reason.py # Extract target-guided behavior(knowledge-based transition) data
python preprocess/filter_embeding.py # Extract the word vectors
```

The processed files are placed in the `data` folder

## Training

### Stage 1 Learning

To train basemodel, please run the following script:

Train on OTTers

```bash
# train a predictor
python code/predictor_OTTers.py
# finetune
python code/keyword_generator.py --dataset ott
# run predict
python code/keyword_generator.py --dataset ott --key_model version_0
```

Ablation global plan and large graph

```bash
python code/predictor_OTTers.py --plan_type 'no_plan'
python code/predictor_OTTers.py --plan_type 'large_graph'
```

Train on TGConv

```bash
# train a predictor
python code/predictor_TGConv.py
# finetune
python code/keyword_generator.py --dataset tgconv
# run predict
python code/keyword_generator.py --dataset tgconv --run_predict version_0 --key_model version_0
```

### Stage 2 Learning

Train a discriminator for reward evalution, please run the following script:

```bash
python code/discriminator.py
```

Finetune the predictor by RL, please run the following script:

Make sure that the discriminator, predictor, and generator paths used in the code are correct

```bash
python code/stage2_ppo.py --base_path logs_base/version_0/checkpoints/best.ckpt --disc_path logs_discri/version_0
```

## Test

The best running results have been saved in the file `output/simulation_output.txt`

```bash
python code/rl_test.py --target easy
python code/rl_test.py --target hard
```

## Citation
If you find our work useful for your research, please kindly cite our paper as follows:
```
@inproceedings{yang-etal-2022-topkg,
    title = "{T}op{KG}: Target-oriented Dialog via Global Planning on Knowledge Graph",
    author = "Yang, Zhitong  and
      Wang, Bo  and
      Zhou, Jinfeng  and
      Tan, Yue  and
      Zhao, Dongming  and
      Huang, Kun  and
      He, Ruifang  and
      Hou, Yuexian",
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    month = oct,
    year = "2022",
    address = "Gyeongju, Republic of Korea",
    publisher = "International Committee on Computational Linguistics",
    url = "https://aclanthology.org/2022.coling-1.62",
    pages = "745--755"
}
```
