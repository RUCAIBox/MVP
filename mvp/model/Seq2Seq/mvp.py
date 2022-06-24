import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from mvp.model.abstract_generator import Seq2SeqGenerator
from transformers import MvpForConditionalGeneration, MvpTokenizerFast

class MVP(Seq2SeqGenerator):

    def __init__(self, config, dataset):
        super(MVP, self).__init__(config, dataset)
        self.pretrained_model_path = config['pretrained_model_path']
        self.dataset_name = config['dataset']

        kwargs = {}
        if config['use_prompt'] is not None:
            kwargs['use_prompt'] = config['use_prompt']
        if config['lightweight_tuning'] is not None:
            kwargs['lightweight_tuning'] = config['lightweight_tuning']
        self.model = MvpForConditionalGeneration.from_pretrained(self.pretrained_model_path, **kwargs)
        self.tokenizer = MvpTokenizerFast.from_pretrained(self.pretrained_model_path)
        if self.dataset_name == 'multiwoz':
            self.tokenizer.add_tokens(['[value_type]', '[value_departure]', '[hotel]', '[offerbooked]', '[value_food]', '[hospital]', '[value_stars]', '[value_arrive]', '[recommend]', '[value_people]', '[request]', '[welcome]', '[value_postcode]', '[value_leave]', '[inform]', '[general]', '[value_reference]', '[bye]', '[value_day]', '[attraction]', '[db_nores]', '[value_car]', '[value_stay]', '[train]', '[db_2]', '[taxi]', '[nooffer]', '[value_choice]', '[value_phone]', '[db_3]', '[nobook]', '[value_area]', '[restaurant]', '[value_pricerange]', '[value_time]', '[value_destination]', '[reqmore]', '[offerbook]', '[greet]', '[value_department]', '[select]', '[value_price]', '[value_address]', '[value_id]', '[police]', '[db_1]', '[value_name]', '[db_0]'])
            
        self.configuration = self.model.config
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.prefix = config['prefix_prompt'] or ''
        self.suffix = config['suffix_prompt'] or ''
        self.truncate = config['truncate'] or 'tail'
        self.prefix_ids = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(self.suffix, add_special_tokens=False)

        self.min_length = config['min_length'] or 1
        self.length_penalty = config['length_penalty'] or 1.0
        self.beam_size = config['beam_size'] or 5
        self.label_smoothing = config['label_smoothing'] if config['label_smoothing'] is not None else 0.1

    def generate(self, batch_data, eval_data):
        input_ids, attn_masks = self._generate_eval_inputs(batch_data)
        if self.dataset_name not in ['roc', 'wp']:
            sample_outputs = self.model.generate(
                input_ids, attention_mask=attn_masks, num_beams=self.beam_size, no_repeat_ngram_size=3, length_penalty=self.length_penalty, min_length=self.min_length, max_length=self.target_max_length, early_stopping=True
            )
        else:
            sample_outputs = self.model.generate(
                input_ids, attention_mask=attn_masks, num_beams=1, do_sample=True, top_p=0.9, temperature=0.7, no_repeat_ngram_size=3, length_penalty=self.length_penalty, min_length=self.min_length, max_length=self.target_max_length
            )
        generated_text = self.tokenizer.batch_decode(sample_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        generate_corpus = [text.strip().split() for text in generated_text]
        generate_corpus = [text or ['none'] for text in generate_corpus]
        return generate_corpus

    def _generate_eval_inputs(self, corpus):
        source_text = corpus['source_text']
        input_ids = []
        attn_masks = []
        src_ids_num = self.source_max_length - len(self.prefix_ids) - len(self.suffix_ids) - self.tokenizer.num_special_tokens_to_add()
        for src_text in source_text:
            src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
            if self.truncate == 'tail':
                src_ids = src_ids[:src_ids_num]
            else:
                src_ids = src_ids[-src_ids_num:]
            input_id = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(self.device)
        return input_ids, attn_masks

    def _generate_default_inputs(self, corpus):
        source_text = corpus['source_text']
        target_text = corpus['target_text']
        input_ids = []
        labels = []
        attn_masks = []
        src_ids_num = self.source_max_length - len(self.prefix_ids) - len(self.suffix_ids) - self.tokenizer.num_special_tokens_to_add()
        tgt_ids_num = self.target_max_length - self.tokenizer.num_special_tokens_to_add()
        for src_text, tgt_text in zip(source_text, target_text):
            src_ids = self.tokenizer.encode(src_text, add_special_tokens=False)
            tgt_ids = self.tokenizer.encode(tgt_text, add_special_tokens=False)
            if self.truncate == 'tail':
                src_ids = src_ids[:src_ids_num]
                tgt_ids = tgt_ids[:tgt_ids_num]
            else:
                src_ids = src_ids[-src_ids_num:]
                tgt_ids = tgt_ids[-tgt_ids_num:]
            input_id = self.tokenizer.build_inputs_with_special_tokens(self.prefix_ids + src_ids + self.suffix_ids)
            label = self.tokenizer.build_inputs_with_special_tokens(tgt_ids)
            input_ids.append(torch.tensor(input_id, dtype=torch.long))
            attn_masks.append(torch.ones(len(input_id), dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0).to(self.device)
        labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(self.device)
        inputs = {'input_ids': input_ids, 'attention_mask': attn_masks, 'labels': labels}
        return inputs

    def forward(self, corpus, epoch_idx=-1):
        inputs = self._generate_default_inputs(corpus)
        outputs = self.model(**inputs)
        if self.label_smoothing:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            return loss_fct(outputs.logits.view(-1, self.configuration.vocab_size), inputs['labels'].view(-1))
        else:
            return outputs.loss
