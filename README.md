# MVP: Multi-task Supervised Pre-training for Natural Language Generation

This repository is the official implementation of our paper [https://arxiv.org/abs/2206.12131](https://arxiv.org/abs/2206.12131). The implementation is completely based on our text generation library **[TextBox 2.0](https://github.com/RUCAIBox/TextBox)**.

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

## Installation
You should clone the TextBox repository and follow its [instructions](https://github.com/RUCAIBox/TextBox#installation).
```bash
git clone https://github.com/RUCAIBox/TextBox.git && cd TextBox
bash install.sh
```

## Datasets

You can download our datasets for fine-tuning in: [https://huggingface.co/RUCAIBox](https://huggingface.co/RUCAIBox). You should create a folder `dataset` and download dataset such as `cnndm` in it.

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
- Paraphrase generation: Quora (quora).
- Text style transfer: GYAFC-E&M and F&R (gyafc_em, gyafc_fr).

## Fine-tuning, Inference and Evaluation

After downloading the dataset, our code can conduct fine-tuning, inference and evaluation in a pipeline.

We propose MVP, MVP+S/M, Single, and BART in our paper, details can be found [here](https://arxiv.org/abs/2206.12131).

### Fine-tuning with MVP:

```bash
python run_textbox.py --model=MVP --dataset=[dataset_name] --model_path=RUCAIBox/mvp
```

`dataset_name` can be one of the name under `dataset` folder, such as `cnndm` and `webnlg`.

### Fine-tuning with MVP+S/M:

```bash
python run_textbox.py --model=MVP --dataset=[dataset_name] --model_path=RUCAIBox/mvp-[task_name]
```

`task_name` can be selected from `summarization`, `open-dialog`, `data-to-text`, `question-generation`, `story`, `question-answering` and `task-dialog`. If you want to fine-tune MVP+M, the `task_name ` should be `multi-task`.

For example, to fine-tune `squadqg` dataset on question generation using MVP+S:

```bash
python run_textbox.py --model=MVP --dataset=squadqg --model_path=RUCAIBox/mvp-question-generation
```

### Fine-tuning with Single and BART:

```bash
python run_textbox.py --model=MVP --dataset=[dataset_name] --model_path=RUCAIBox/mtl-[task_name]
```

`task_name` can be selected from `summarization`, `open-dialog`, `data-to-text`, `question-generation`, `story`, `question-answering` and `task-dialog`.

We also support to fine-tune with BART:

```bash
python run_textbox.py --model=BART --dataset=[dataset_name] --model_path=facebook/bart-large
```

### Lightweight Tuning:

If you want to conduct lightweight tuning of MVP+S/M, just add the option `--lightweight_tuning=True` in the script.

For example, to lightweight tune `roc` dataset using MVP+M:

```bash
python run_textbox.py --model=MVP --dataset=roc --model_path=RUCAIBox/mvp-multi-task --lightweight_tuning=True
```

We also support to lightweight tune with BART+R (*i.e.,* Prefix-tuning) [here](https://github.com/RUCAIBox/TextBox#parameter-efficient-prompting).


## Citation
```bibtex
@article{tang2022mvp,
  title={MVP: Multi-task Supervised Pre-training for Natural Language Generation},
  author={Tang, Tianyi and Li, Junyi and Zhao, Wayne Xin and Wen, Ji-Rong},
  journal={arXiv preprint arXiv:2206.12131},
  year={2022},
  url={https://arxiv.org/abs/2206.12131},
}
```
