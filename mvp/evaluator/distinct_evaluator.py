import numpy as np
from collections import defaultdict, Counter
from nltk import word_tokenize
from mvp.evaluator.abstract_evaluator import AbstractEvaluator


class DistinctEvaluator(AbstractEvaluator):
    r"""Distinct Evaluator. Now, we support metrics `'inter-distinct'` and `'intra-distinct'`.
    """

    def __init__(self):
        self.n_grams = [1, 2, 3, 4]

    def dist_func(self, generate_sentence, ngram):
        ngram_dict = defaultdict(int)
        tokens = generate_sentence[:ngram]
        ngram_dict[" ".join(tokens)] += 1
        for i in range(1, len(generate_sentence) - ngram + 1):
            tokens = tokens[1:]
            tokens.append(generate_sentence[i + ngram - 1])
            ngram_dict[" ".join(tokens)] += 1
        return ngram_dict

    def _calc_metrics_info(self, generate_corpus, reference_corpus=None):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus

        Returns:
            list: a list of metrics <metric> which record the results
        """
        dist_dict = {}
        intra_ngrams = []
        for n_gram in self.n_grams:
            ngrams_all = Counter()
            intra_ngram = []
            for generate_sentence in generate_corpus:
                result = self.dist_func(generate_sentence=generate_sentence, ngram=n_gram)
                intra_ngram.append(len(result) / sum(result.values()))
                ngrams_all.update(result)
            key = "distinct-{}".format(n_gram)
            dist_dict[key] = len(ngrams_all) / sum(ngrams_all.values())
            intra_ngrams.append(np.average(intra_ngram))
        return dist_dict
