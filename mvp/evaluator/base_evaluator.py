from mvp.evaluator.bleu_evaluator import BleuEvaluator
from mvp.evaluator.distinct_evaluator import DistinctEvaluator
from mvp.evaluator.meteor_evaluator import MeteorEvaluator
from mvp.evaluator.rouge_evaluator import RougeEvaluator
from mvp.evaluator.squad_evaluator import SQuADEvaluator
from mvp.evaluator.coqa_evaluator import CoQAEvaluator
from mvp.evaluator.dt_evaluator import DTEvaluator
from mvp.evaluator.gem_evaluator import GEMEvaluator
from mvp.evaluator.da_evaluator import DAEvaluator
from mvp.evaluator.qa_evaluator import QAEvaluator


evaluator_list = [
    'bleu', 'rouge', 'distinct', 'meteor', 'squad', 'coqa', 'dt', 'gem', 'da', 'qa'
]


class BaseEvaluator():

    def __init__(self, config, metrics):
        self.config = config
        self.metrics = metrics
        # [1, 2, 3, 4]

    def evaluate(self, generate_corpus, reference_corpus):
        r"""get metrics result

        Args:
            generate_corpus: the generated corpus
            reference_corpus: the referenced corpus
        
        Returns:
            dict: such as ``{'bleu-1': xxx, 'bleu-1-avg': yyy}``
        """
        result_dict = {}
        for metric in self.metrics:
            if metric == 'bleu':
                task_type = (self.config['task_type'].lower() == "unconditional")
                evaluator = BleuEvaluator(task_type, self.config['smoothing_function'])
            elif metric == 'rouge':
                evaluator = RougeEvaluator()
            elif metric == 'distinct':
                evaluator = DistinctEvaluator()
            elif metric == 'meteor':
                evaluator = MeteorEvaluator()
            elif metric == 'squad':
                evaluator = SQuADEvaluator(self.config)
            elif metric == 'coqa':
                evaluator = CoQAEvaluator()
            elif metric == 'dt':
                evaluator = DTEvaluator(self.config)
            elif metric == 'gem':
                evaluator = GEMEvaluator()
            elif metric == 'da':
                evaluator = DAEvaluator()
            elif metric == 'qa':
                evaluator = QAEvaluator()
            metric_result = evaluator.evaluate(generate_corpus=generate_corpus, reference_corpus=reference_corpus)
            result_dict.update(metric_result)
        return result_dict
