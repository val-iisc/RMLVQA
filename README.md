## Learnable Adaptive Margin Loss To Overcome Language Bias in Visual Question Answering

This repository contains the implementation of our model AdaArc-LM.
This repository is built upon https://github.com/guoyang9/AdaVQA. 

Almost all flags can be set at `utils/config.py`. The dataset paths, the hyperparams can be set accordingly in this 
file.

## GPU used: 
	* One NVIDIA GeForce RTX 2080 Tis
	
## Memory required:
	* 4GB approximately

## Prerequisites
    * python==3.7.11
    * nltk==3.7
    * bcolz==1.2.1
    * tqdm==4.62.3
    * numpy==1.21.4  
    * pytorch==1.10.2
    * tensorboardX==2.4
    * torchvision==0.11.3
    * h5py==3.5.0

## Dataset

* Download the VQA-CP datasets from the link provided in the supplementary material.
* The image features can be downloaded by following instructions from : https://github.com/hengyuan-hu/bottom-up-attention-vqa.
* The pre-trained Glove features can be accessed via https://nlp.stanford.edu/projects/glove/.

After downloading the datasets, keep them in the folders set by config.py

## Preprocessing

The preprocessing steps are as follows:

1. process questions and dump dictionary:
    ```
    python tools/create_dictionary.py
    ```

2. process answers and question types, and generate the frequency-based margins:
    ```
    python tools/compute_softscore.py
    ```
3. convert image features to h5:
    ```
    python tools/detection_features_converter.py 
    ```

## Model training instruction
```
    python main_arcface.py --name test-VQA --gpu 0
   ```
## Model evaluation instruction
```
    python main_arcface.py --name test-VQA --eval-only
   ```
Running this code creates a new json file (eg. abc.json), which contains test question ids and the
answers predicted by the model.

## Category wise evaluation instruction
```
python acc_per_type.py abc.json
```
The argument name refers to the name of the file in which the model weights will be finally stored.

## Results on AdaArc and AdaArc-LM evaluated on VQA-CP v2

| Model             		   | Accuracy  in %|
|:----------------------------:|:-------------:|
| AdaArc                       | 57.24         |
| + Randomization              | 57.97         |
| +Bias-injection              | 59.44         |
| +Learnable margins           | 59.87         |
| +Supervised Conctrastive Loss| 60.41         |
