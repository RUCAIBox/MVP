import numpy as np
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
from mvp.evaluator.abstract_evaluator import AbstractEvaluator


class BleuEvaluator(AbstractEvaluator):
    r"""Bleu Evaluator. Now, we support metrics `'bleu'`
    """

    def __init__(self, task_type, smoothing_function):
        self.n_grams = [1, 2, 3, 4]
        self.task_type = task_type
        self.smoothing_function = smoothing_function or 1
        self.weights = self._generate_weights()

    def _generate_weights(self):
        weights = {}
        for n_gram in self.n_grams:
            avg_weight = [1. / n_gram] * n_gram
            avg_weight.extend([0. for index in range(max(self.n_grams) - n_gram)])
            weights['bleu-{}'.format(n_gram)] = tuple(avg_weight)
        return weights

    def _calc_metrics_info(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus

        Returns:
            dict: a dict of metrics <metric> which record the results according to self.n_grams
        """

        bleu_dict = {}
        for n_gram in self.n_grams:
            bleu_dict['bleu-{}'.format(n_gram)] = []

        if isinstance(self.smoothing_function, int):
            for i in range(len(generate_corpus)):
                pred_sent = generate_corpus[i]
                gold_sent = reference_corpus[i]
                for n_gram in self.n_grams:
                    results = sentence_bleu(
                        hypothesis=pred_sent,
                        references=[gold_sent],
                        weights=self.weights['bleu-{}'.format(n_gram)],
                        smoothing_function=getattr(SmoothingFunction(), f"method{self.smoothing_function}")
                    )
                    bleu_dict['bleu-{}'.format(n_gram)].append(results)
        else:
            for n_gram in self.n_grams:
                results = corpus_bleu(
                    [[r] for r in reference_corpus],
                    generate_corpus,
                    weights=self.weights['bleu-{}'.format(n_gram)],
                )
                bleu_dict['bleu-{}'.format(n_gram)].append(results)
        return bleu_dict
