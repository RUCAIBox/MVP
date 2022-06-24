import os
from mvp.data.dataset import AbstractDataset
from mvp.data.utils import load_data


class PairedSentenceDataset(AbstractDataset):

    def __init__(self, config):
        self.share_vocab = config['share_vocab']
        super().__init__(config)

    def _get_preset(self):
        super()._get_preset()
        self.source_text = []
        self.target_text = []

    def _load_source_data(self):
        for i, prefix in enumerate(['train', 'valid', 'test']):
            filename = os.path.join(self.dataset_path, f'{prefix}.src')
            text_data = load_data(
                filename, self.tokenize_strategy, self.source_max_length, self.source_language,
                self.source_multi_sentence, self.source_max_num
            )
            if self.config['metrics'] != ['dt']:
                assert len(text_data) == len(self.target_text[i])
            self.source_text.append(text_data)
