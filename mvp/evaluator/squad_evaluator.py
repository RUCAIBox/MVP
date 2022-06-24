import numpy as np
import os
from mvp.evaluator.abstract_evaluator import AbstractEvaluator


class SQuADEvaluator(AbstractEvaluator):

    def __init__(self, config):
        super().__init__()
        self.saved_text_file = os.path.join(config['generated_text_dir'], config['filename'] + '.txt')
        self.dataset = config['dataset']

    def evaluate(self, generate_corpus, reference_corpus):
        res = os.popen(f'python2 mvp/evaluator/qg/eval.py -out {self.saved_text_file}  -src dataset/{self.dataset}/test.src -tgt dataset/{self.dataset}/test.tgt')
        res = eval(res.read())
        return res
