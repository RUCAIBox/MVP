import os
from posix import listdir
import nltk
import collections
import torch
import copy
import shutil
from logging import getLogger


def get_dataset(config):
    """Create dataset according to :attr:`config['model']` and :attr:`config['MODEL_TYPE']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    from .dataset import PairedSentenceDataset
    return PairedSentenceDataset


def dataloader_construct(name, config, dataset, batch_size=1, shuffle=False, drop_last=True):
    """Get a correct dataloader class by calling :func:`get_dataloader` to construct dataloader.

    Args:
        name (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset or list of Dataset): The split dataset for constructing dataloader.
        batch_size (int, optional): The batch_size of dataloader. Defaults to ``1``.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
        drop_last (bool, optional): Whether the dataloader will drop the last batch. Defaults to ``True``.

    Returns:
        AbstractDataLoader or list of AbstractDataLoader: Constructed dataloader in split dataset.
    """

    task_type = config['task_type'].lower()
    logger = getLogger()
    logger.info('Build [{}] DataLoader for [{}]'.format(task_type, name))
    logger.info('batch_size = [{}], shuffle = [{}], drop_last = [{}]\n'.format(batch_size, shuffle, drop_last))

    DataLoader = get_dataloader(config)

    return DataLoader(
        config=config, dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last
    )


def get_dataloader(config):
    """Return a dataloader class according to :attr:`config` and :attr:`split_strategy`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`split_strategy`.
    """
    from .dataloader import PairedSentenceDataLoader
    return PairedSentenceDataLoader


def construct_quick_test_dataset(dataset_path):
    files = listdir(dataset_path)
    for file in files:
        filename = os.path.join(dataset_path, file)
        if filename.endswith('.bin'):
            os.remove(filename)
        else:
            shutil.copy(filename, filename + '.tmp')
    for file in files:
        filename = os.path.join(dataset_path, file)
        if not filename.endswith('.bin'):
            with open(filename + '.tmp', 'r') as fin, open(filename, 'w') as fout:
                for line in fin.readlines()[:10]:
                    fout.write(line)


def deconstruct_quick_test_dataset(dataset_path):
    files = listdir(dataset_path)
    for file in files:
        filename = os.path.join(dataset_path, file)
        if filename.endswith('.bin'):
            os.remove(filename)
        elif not filename.endswith('.tmp'):
            shutil.move(filename + '.tmp', filename)


def data_preparation(config, save=False):
    """call :func:`dataloader_construct` to create corresponding dataloader.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        save (bool, optional): If ``True``, it will call :func:`save_datasets` to save split dataset.
            Defaults to ``False``.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    if config['quick_test']:
        construct_quick_test_dataset(config['data_path'])
    dataset = get_dataset(config)(config)
    if config['quick_test']:
        deconstruct_quick_test_dataset(config['data_path'])

    train_dataset = copy.copy(dataset)
    valid_dataset = copy.copy(dataset)
    test_dataset = copy.copy(dataset)
    for prefix in ['train', 'valid', 'test']:
        dataset = locals()[f'{prefix}_dataset']
        content = getattr(dataset, f'{prefix}_data')
        for key, value in content.items():
            setattr(dataset, key, value)

    train_data = dataloader_construct(
        name='train',
        config=config,
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
    )

    valid_data = dataloader_construct(
        name='valid',
        config=config,
        dataset=valid_dataset,
        batch_size=config['eval_batch_size'],
        drop_last=False
    )

    test_data = dataloader_construct(
        name='test',
        config=config,
        dataset=test_dataset,
        batch_size=config['eval_batch_size'],
        drop_last=False,
    )

    return train_data, valid_data, test_data


def tokenize(text, tokenize_strategy, language, multi_sentence):
    """Tokenize text data.

    Args:
        text (str): text data.
        tokenize_strategy (str): strategy of tokenizer.
        language (str): language of text.
        multi_sentence (bool): whether to split text into sentence level.
    
    Returns:
        List[str]: the tokenized text data.
    """
    if multi_sentence:
        text = text.split('\t')
        if tokenize_strategy == 'none':
            words = text
        elif tokenize_strategy == 'by_space':
            words = [t.split() for t in text]
        elif tokenize_strategy == 'nltk':
            words = [nltk.word_tokenize(t, language=language) for t in text]
    else:
        text.replace('\t', ' ')
        if tokenize_strategy == 'none':
            words = text
        elif tokenize_strategy == 'by_space':
            words = text.split()
        elif tokenize_strategy == 'nltk':
            words = nltk.word_tokenize(text, language=language)
    return words


def load_data(dataset_path, tokenize_strategy, max_length, language, multi_sentence, max_num):
    """Load dataset from split (train, valid, test).
    This is designed for single sentence format.

    Args:
        dataset_path (str): path of dataset dir.
        tokenize_strategy (str): strategy of tokenizer.
        max_length (int): max length of sequence.
        language (str): language of text.
        multi_sentence (bool): whether to split text into sentence level.
        max_num (int): max number of sequence.
    
    Returns:
        List[List[str]]: the text list loaded from dataset path.
    """
    if not os.path.isfile(dataset_path):
        raise ValueError('File {} not exist'.format(dataset_path))

    text = []
    with open(dataset_path, "r") as fin:
        for line in fin:
            line = line.strip()
            words = tokenize(line, tokenize_strategy, language, multi_sentence)
            if isinstance(words, str):  # no split
                text.append(words)
            elif isinstance(words[0], str):  # single sentence
                text.append(words[:max_length])
            else:  # multiple sentences
                text.append([word[:max_length] for word in words[:max_num]])
    return text

def pad_sequence(idx, length, padding_idx, num=None):
    r"""padding a batch of word index data, to make them have equivalent length

    Args:
        idx (List[List[int]] or List[List[List[int]]]): word index
        length (List[int] or List[List[int]]): sequence length
        padding_idx (int): the index of padding token
        num (List[int]): sequence number
    
    Returns:
        idx (List[List[int]] or List[List[List[int]]]): word index
        length (List[int] or List[List[int]]): sequence length
        num (List[int]): sequence number
    """
    if num is None:
        max_length = max(length)
        new_idx = []
        for sent_idx, sent_length in zip(idx, length):
            new_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
        new_idx = torch.LongTensor(new_idx)
        length = torch.LongTensor(length)
        return new_idx, length, None
    else:
        max_length = max([max(sent_length) for sent_length in length])
        max_num = max(num)
        new_length = []
        new_idx = []
        for doc_idx, doc_length, doc_num in zip(idx, length, num):
            new_length.append(doc_length + [0] * (max_num - doc_num))
            new_sent_idx = []
            for sent_idx, sent_length in zip(doc_idx, doc_length):
                new_sent_idx.append(sent_idx + [padding_idx] * (max_length - sent_length))
            for _ in range(max_num - doc_num):
                new_sent_idx.append([0] * max_length)
            new_idx.append(new_sent_idx)

        new_num = torch.LongTensor(num)
        new_length = torch.LongTensor(new_length)
        new_idx = torch.LongTensor(new_idx)
        return new_idx, new_length, new_num
