import numpy as np
import sacrebleu
from nltk import word_tokenize
import numpy as np
from itertools import zip_longest
from rouge_score import rouge_scorer, scoring
from mvp.evaluator.abstract_evaluator import AbstractEvaluator
from mvp.evaluator.meteor import PyMeteorWrapper


class GEMEvaluator(AbstractEvaluator):

    def evaluate(self, generate_corpus, reference_corpus):
        metric = {}
        m = PyMeteorWrapper('en')
        generate_corpus = [' '.join(g) for g in generate_corpus]
        if isinstance(reference_corpus[0][0], str) or (isinstance(reference_corpus[0], list) and len(reference_corpus[0]) == 1):
            if (isinstance(reference_corpus[0], list) and len(reference_corpus[0]) == 1):
                reference_corpus = [r[0] for r in reference_corpus]
            reference_corpus = [' '.join(r) for r in reference_corpus]
            metric['bleu'] = round(sacrebleu.corpus_bleu(generate_corpus, [reference_corpus], lowercase=True).score, 2)
            reference_corpus = [[r] for r in reference_corpus]
            metric['meteor'] = round(m.compute_score(generate_corpus, reference_corpus)[0] * 100, 2)
            reference_corpus = [' '.join(word_tokenize(r[0])) for r in reference_corpus]
        else:
            reference_corpus = [[' '.join(r) for r in ref] for ref in reference_corpus]
            metric['bleu'] = round(sacrebleu.corpus_bleu(generate_corpus, list(zip_longest(*reference_corpus)), lowercase=True).score, 2)
            metric['meteor'] = round(m.compute_score(generate_corpus, reference_corpus)[0] * 100, 2)
            reference_corpus = [[' '.join(word_tokenize(r)) for r in ref] for ref in reference_corpus]
        scores = []
        rouge = rouge_scorer.RougeScorer(rouge_types=["rouge2"], use_stemmer=True)
        for refs, pred in zip(reference_corpus, generate_corpus):
            if isinstance(refs, str):
                score = rouge.score(refs, pred)['rouge2'].fmeasure
            else:
                cur_scores = [rouge.score(ref, pred) for ref in refs]
                best_scores = []
                for leave in range(len(refs)):
                    cur_scores_leave_one = [cur_scores[s] for s in range(len(refs)) if s != leave]
                    best_scores.append(max([s['rouge2'] for s in cur_scores_leave_one], key=lambda s: s.fmeasure))
                score = np.mean(best_scores)
            scores.append(score)
        metric['rouge'] = round(np.mean(scores) * 100, 2)
        return metric
