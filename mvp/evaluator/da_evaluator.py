import numpy as np
import os
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from mvp.evaluator.abstract_evaluator import AbstractEvaluator

def score_fn(_hyp: dict, _ref: dict):
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    _scores = {}
    for scorer, method in scorers:
        _score, _ = scorer.compute_score(_ref, _hyp)
        if type(_score) == list:
            for m, s in zip(method, _score):
                _scores[m] = round(s * 100, 2)
        else:
            _scores[method] = round(_score * 100, 2) if method != 'CIDEr' else round(_score, 3)
    return _scores

class DAEvaluator(AbstractEvaluator):

    def evaluate(self, generate_corpus, reference_corpus):
        reference_corpus = [' '.join(r).split(' | ') for r in reference_corpus]
        generate_corpus = [' '.join(r) for r in generate_corpus]
        hyps = {idx: [hyp] for idx, hyp in enumerate(generate_corpus)}
        refs = {idx: ref for idx, ref in enumerate(reference_corpus)}
        res = score_fn(hyps, refs)
        return res
