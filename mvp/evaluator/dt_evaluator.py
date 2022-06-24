import numpy as np
import os
from mvp.evaluator.abstract_evaluator import AbstractEvaluator


class DTEvaluator(AbstractEvaluator):

    def __init__(self, config):
        super().__init__()
        self.saved_text_file = os.path.join(config['generated_text_dir'], config['filename'] + '.txt')
        self.dataset = config['dataset']

    def evaluate(self, generate_corpus, reference_corpus):
        res = os.popen(f'python mvp/evaluator/d2t/measure_scores.py dataset/{self.dataset}/test.tgt {self.saved_text_file}')
        res = eval(res.read())
        return res
