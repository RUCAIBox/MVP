import numpy as np
import torch
import torch.nn as nn

from mvp.utils import ModelType


class AbstractModel(nn.Module):
    r"""Base class for all models
    """

    def __init__(self, config, dataset):
        # load parameters info
        super(AbstractModel, self).__init__()
        self.device = config['device']
        self.batch_size = config['train_batch_size']
        self.dataset = dataset

    def __getattr__(self, name):
        if hasattr(self.dataset, name):
            value = getattr(self.dataset, name)
            if value is not None:
                return value
        return super().__getattr__(name)

    def generate(self, batch_data, eval_data):
        r"""Predict the texts conditioned on a noise or sequence.

        Args:
            batch_data (Corpus): Corpus class of a single batch.
            eval_data: Common data of all the batches.

        Returns:
            torch.Tensor: Generated text, shape: [batch_size, max_len]
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)


class Seq2SeqGenerator(AbstractModel):
    """This is a abstract general seq2seq generator. All the seq2seq model should implement this class.
    The base general seq2seq generator class provide the basic parameters information.
    """
    type = ModelType.SEQ2SEQ

    def __init__(self, config, dataset):
        super(Seq2SeqGenerator, self).__init__(config, dataset)