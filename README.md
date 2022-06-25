# MVP: Multi-task Supervised Pre-training for Natural Language Generation

This repository is the official implementation of our paper.

## Overview

- MVP follows a standard Transformer encoder-decoder architecture.
- MVP is supervised pre-trained using labeled datasets.
- MVP also has task-specific soft prompts to stimulate the model's capacity in performing a certain task.
- MVP is specially designed for natural language generation and can be adapted to a wide range of generation tasks. Our model can also be adapted to natural language understanding tasks.

![model](model.jpg)

Tips:

- We have released a series of models in [HuggingFace](https://huggingface.co/models?filter=mvp), including MVP, MVP with task-specific prompts, and multi-task pre-trained variants.
- If you want to use a model without prompts, you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp')`.
- If you want to use a model with task-specific prompts, such as summarization, you can load it through `MvpForConditionalGeneration.from_pretrained('RUCAIBox/mvp-summarization')`.
- Our model supports lightweight prompt tuning following [Prefix-tuning](https://arxiv.org/abs/2101.00190) with config `lightweight_tuning=True`.

## Requirements

To install requirements:

```bash
bash install.sh
```

## Datasets

You can download our datasets for fine-tuning in: [https://drive.google.com/drive/folders/1ZJdXrBhBPtfozPqhIv-Yid9yemXRJiZQ?usp=sharing](https://drive.google.com/drive/folders/1ZJdXrBhBPtfozPqhIv-Yid9yemXRJiZQ?usp=sharing). You should create a folder `dataset` and download dataset such as `cnndm` in it.

Now we support 11 generation tasks and corresponding datasets:
- Text summarization: CNN/Daily Mail (cnndm), XSum (xsum), SAMSum (samsum), and WLE (wle).
- Open-ended dialogue system: PersonaChat (pc), DailyDialog (dd), DSTC7-AVSD (da), and SGD (sgd).
- Data-to-text generation: WebNLG v2.1 (webnlg), WebNLG v3.0 (webnlg2), WikiBio (wikibio), E2E (e2e), DART (dart), and ToTTo (totto).
- Question generation: SQuAD (squadqg) and CoQA (coqaqg).
- Story generation: ROCStories (roc) and WritingPrompts (wp).
- Question answering: SQuAD (squad) and CoQA (coqa).
- Task-oriented dialogue system: MultiWOZ 2.0 (multiwoz).
- Commonsense generation: CommonGen (cg).
- Text simplification: WikiAuto + Turk/ASSET (wia).
- Paraphrase generation: Quora (comming soon).
- Text style transfer: GYAFC-E&M and F&R (comming soon).

## Fine-tuning, Inference and Evaluation

After downloading the dataset in this folder, our code can conduct fine-tuning, inference and evaluation in a pipeline.

We propose MVP, Single, MVP+R, MVP+S and MVP+M in our paper, details can be found [here](). The latter three support lightweight prompt tuning.

### Fine-tuning with MVP:

```bash
python run.py --dataset=[dataset_name]
```

`dataset_name` can be one of the name under `dataset` folder, such as `cnndm` and `webnlg`.

### Fine-tuning with Single:

```bash
python run.py --dataset=[dataset_name] --pretrained_model_path=RUCAIBox/mtl-[task_name]
```

`task_name` can be selected from `summarization`, `open-dialog`, `data-to-text`, `question-generation`, `story`, `question-answering` and `task-dialog`.

We also support to fine-tune with BART:

```bash
python run.py --dataset=[dataset_name] --pretrained_model_path=facebook/bart-large
```

### Fine-tuning with MVP+R:

```bash
python run.py --dataset=[dataset_name] --use_prompt=True
```

We also support to fine-tune with BART+R:

```bash
python run.py --dataset=[dataset_name] --pretrained_model_path=facebook/bart-large --use_prompt=True
```

### Fine-tuning with MVP+S/M:

```bash
python run.py --dataset=[dataset_name] --pretrained_model_path=RUCAIBox/mvp-[task_name]
```

`task_name` can be selected from `summarization`, `open-dialog`, `data-to-text`, `question-generation`, `story`, `question-answering` and `task-dialog`. If you want to fine-tune MVP+M, the `task_name ` should be `multi-task`.

For example, to fine-tune `squadqg` dataset on question generation using MVP+S:

```bash
python run.py --dataset=squadqg --pretrained_model_path=RUCAIBox/mvp-question-generation
```

### Lightweight Tuning:

If you want to conduct lightweight tuning of MVP+R/S/M, just add the option `--lightweight_tuning=True` in the script.

For example, to lightweight tune `roc` dataset using MVP+M:

```bash
python run.py --dataset=roc --pretrained_model_path=RUCAIBox/mvp-multi-task --use_prompt=True --lightweight_tuning=True
```

We also support to lightweight tune with BART+R (*i.e.,* Prefix-tuning):

```bash
python run.py --dataset=[dataset_name] --pretrained_model_path=facebook/bart-large --use_prompt=True --lightweight_tuning=True
```

## Intermediate Results
The generated texts of our models on each dataset is coming soon.
