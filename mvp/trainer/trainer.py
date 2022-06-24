import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import collections

from tqdm import tqdm
from nltk import word_tokenize
from torch.utils.data import DataLoader
from time import time
from logging import getLogger

from mvp.evaluator import BaseEvaluator, evaluator_list
from mvp.utils import ensure_dir, early_stopping


class AbstractTrainer(object):
    def __init__(self, config, model):
        self.config = config
        self.model = model

class Trainer(AbstractTrainer):

    def __init__(self, config, model, batch_num):
        super(Trainer, self).__init__(config, model)

        self.logger = getLogger()
        self.learner = config['learner']
        self.learning_rate = config['learning_rate']
        self.epochs = config['epochs']
        self.eval_step = min(config['eval_step'], self.epochs)
        self.stopping_step = config['stopping_step']
        self.test_batch_size = config['eval_batch_size']
        self.device = config['device']
        self.embedding_size = config['embedding_size']
        self.warmup_steps = config['warmup_steps']
        self.checkpoint_dir = config['checkpoint_dir']
        self.grad_clip = config['grad_clip']
        self.update_freq = config['update_freq'] if config['update_freq'] is not None else 1
        self.eval_metrics = config['eval_metrics']
        ensure_dir(self.checkpoint_dir)
        saved_model_file = self.config['filename'] + '.pth'
        self.saved_model_file = os.path.join(self.checkpoint_dir, saved_model_file)

        self.generated_text_dir = config['generated_text_dir']
        ensure_dir(self.generated_text_dir)
        saved_text_file = self.config['filename'] + '.txt'
        self.saved_text_file = os.path.join(self.generated_text_dir, saved_text_file)

        self.start_epoch = 0
        self.cur_step = 0
        self.best_score = -1e10
        self.best_result = None
        self.train_loss_dict = dict()
        self.model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.training_steps = (batch_num // self.update_freq + 1) * self.epochs
        self.optimizer = self._build_optimizer()

        self.metrics = config["metrics"]
        self._check_metrics()
        self.evaluator = BaseEvaluator(config, self.metrics)

        self.item_tensor = None
        self.tot_item_num = None
        self.iid_field = config['ITEM_ID_FIELD']

    def _check_metrics(self):
        r"""check the correct of the setting"""
        if isinstance(self.metrics, (str, list)):
            if isinstance(self.metrics, str):
                if self.metrics[0] == '[':
                    self.metrics = self.metrics[1:]
                if self.metrics[-1] == ']':
                    self.metrics = self.metrics[:-1]
                self.metrics = self.metrics.strip().split(",")
            self.metrics = [metric.lower() for metric in self.metrics]
            for metric in self.metrics:
                if metric not in evaluator_list:
                    raise ValueError(
                        "evaluator {} can't be found. ".format(metric) + "(evaluator should be in [" +
                        ", ".join(evaluator_list) + "])"
                    )
        else:
            raise TypeError('evaluator must be a string or list')

    def _build_optimizer(self):
        r"""Init the Optimizer

        Returns:
            torch.optim: the optimizer
        """
        if self.learner.lower() == 'adamw':
            # optimizer = optim.AdamW(self.model_parameters, lr=self.learning_rate, betas=(0.9, 0.98), eps=1e-06)
            optimizer = optim.AdamW(self.model_parameters, lr=self.learning_rate)
        return optimizer

    def _train_epoch(self, train_data, epoch_idx):
        r"""Train the model in an epoch

        Args:
            train_data (DataLoader): the train data
            epoch_idx (int): the current epoch id

        Returns:
            float/tuple: The sum of loss returned by all batches in this epoch. If the loss in each batch contains
            multiple parts and the model return these multiple parts loss instead of the sum of loss, It will return a
            tuple which includes the sum of loss in each part.
        """
        self.model.train()
        total_loss = None

        pbar = train_data
        pbar = tqdm(pbar, ncols=80)

        step = 0
        self.optimizer.zero_grad()
        for i, data in enumerate(pbar):
            losses = self.model(data, epoch_idx=epoch_idx)
            if isinstance(losses, tuple):
                loss = sum(losses)
                loss_tuple = tuple(per_loss.item() for per_loss in losses)
                total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
            else:
                loss = losses
                total_loss = losses.item() if total_loss is None else total_loss + losses.item()

            self._check_nan(loss)
            pbar.set_postfix(loss=loss.item())
            loss = loss / self.update_freq
            loss.backward()
            step += 1
            if step == self.update_freq:
                torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip)
                self.optimizer.step()
                self.optimizer.zero_grad()
                step = 0
        
        if step:
            torch.nn.utils.clip_grad_norm_(self.model_parameters, self.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad()
        train_loss = total_loss / len(train_data)
        return train_loss

    def _valid_epoch(self, valid_data):
        r"""Valid the model with valid data

        Args:
            valid_data (DataLoader): the valid data

        Returns:
            float: valid score
            dict: valid result
        """
        with torch.no_grad():
            self.model.eval()
            total_loss = None
            for data in tqdm(valid_data, ncols=80):
                losses = self.model(data)
                if isinstance(losses, tuple):
                    loss = sum(losses)
                    loss_tuple = tuple(per_loss.item() for per_loss in losses)
                    total_loss = loss_tuple if total_loss is None else tuple(map(sum, zip(total_loss, loss_tuple)))
                else:
                    loss = losses
                    total_loss = losses.item() if total_loss is None else total_loss + losses.item()
                self._check_nan(loss)
            valid_loss = total_loss / len(valid_data)
            ppl = np.exp(valid_loss)
        return valid_loss, ppl

    def _save_checkpoint(self, epoch, suff=''):
        r"""Store the model parameters information and training information.

        Args:
            epoch (int): the current epoch id

        """
        state = {
            'config': self.config,
            'epoch': epoch,
            'cur_step': self.cur_step,
            'best_score': self.best_score,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        torch.save(state, self.saved_model_file + suff)

    def _save_generated_text(self, generated_corpus):
        r"""Store the generated text by our model.

        Args:
            corpus (list of string list):
        """
        with open(self.saved_text_file, 'w') as fin:
            for tokens in generated_corpus:
                fin.write(' '.join(tokens) + '\n')

    def resume_checkpoint(self, resume_file, flag=False):
        r"""Load the model parameters information and training information.

        Args:
            resume_file (file): the checkpoint file

        """
        resume_file = str(resume_file)
        checkpoint = torch.load(resume_file)
        if not flag:
            self.start_epoch = checkpoint['epoch'] + 1
            self.cur_step = checkpoint['cur_step']
            self.best_score = checkpoint['best_score']
            self.optimizer.load_state_dict(checkpoint['optimizer'])

        # load architecture params from checkpoint
        if checkpoint['config']['model'].lower() != self.config['model'].lower():
            self.logger.warning(
                'Architecture configuration given in config file is different from that of checkpoint. '
                'This may yield an exception while state_dict is being loaded.'
            )
        
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        # load optimizer state from checkpoint only when optimizer type is not changed
        message_output = 'Checkpoint loaded. Resume training from epoch {}'.format(self.start_epoch)
        self.logger.info(message_output)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _generate_train_loss_output(self, epoch_idx, s_time, e_time, losses, train_info=""):
        train_loss_output = "epoch %d %straining [time: %.2fs, " % (epoch_idx, train_info, e_time - s_time)
        if isinstance(losses, tuple):
            for idx, loss in enumerate(losses):
                train_loss_output += 'train_loss%d: %.4f, ' % (idx + 1, loss)
            train_loss_output = train_loss_output[:-2]
        else:
            train_loss_output += "train loss: %.4f" % losses
        return train_loss_output + ']'

    def reduce_loss(self, loss):
        torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.SUM)
        loss /= torch.distributed.get_world_size()
        return loss

    def multiwoz_eval(self):
        import json
        from mvp.evaluator.multiwoz.config import Config
        from mvp.evaluator.multiwoz.reader import MultiWozReader
        from mvp.evaluator.multiwoz.eval import MultiWozEvaluator

        def generate(text, mode):
            input_ids, attn_masks = self.model._generate_eval_inputs({'source_text': text})
            beam_size = self.config['beam_size'] or 5
            output = self.model.model.generate(
                input_ids, attention_mask=attn_masks, max_length=128, num_beams=beam_size, no_repeat_ngram_size=3, early_stopping=True
            )
            generated_text = self.model.tokenizer.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            return generated_text

        def truncate(text):
            input = self.model.tokenizer(text)['input_ids'][1:-1][-1000:]
            text = self.model.tokenizer.decode(input, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            return text

        cfg = Config('mvp/evaluator/multiwoz/multiwoz')
        reader = MultiWozReader(cfg)
        eval = MultiWozEvaluator(reader, cfg)
        with open(f'mvp/evaluator/multiwoz/multiwoz-fine-processed-test.json', 'r') as f:
            data = json.load(f)
        
        datas = []
        for d in tqdm(data, ncols=80):
            prev = ""
            bs_inputs = []
            for t in d:
                for k, v in t.items():
                    if isinstance(v, str) and v.startswith('<sos'):
                        t[k] = v[8:-8].strip()
                bs_input = 'Belief state: [X_SEP] ' + truncate(prev + ' [SEP] ' + t['user'])
                prev += t['user'] + ' [SEP] ' + t['resp']
                bs_inputs.append(bs_input)
            bs_outputs = generate(bs_inputs, 'bs')
            da_inputs = []
            nlg_inputs = []
            prev = ""
            for t, bs_output in zip(d, bs_outputs):
                t['bspn_gen'] = bs_output
                db_text = reader.bspan_to_DBpointer(bs_output, t['turn_domain'])
                da_input = 'Dialogue action: [X_SEP] ' + db_text + ' [X_SEP] ' + truncate(prev + ' [SEP] ' + t['user'])
                nlg_input = 'System response: [X_SEP] ' + db_text + ' [X_SEP] ' + truncate(prev + ' [SEP] ' + t['user'])
                prev += t['user'] + ' [SEP] ' + t['resp']
                
                da_inputs.append(da_input)
                nlg_inputs.append(nlg_input)
            da_outputs = generate(da_inputs, 'da')
            nlg_outputs = generate(nlg_inputs, 'nlg')
            for t, da_output, nlg_output in zip(d, da_outputs, nlg_outputs):
                t['aspn_gen'] = da_output
                t['resp_gen'] = nlg_output
                datas.append(t)
        
        with open('multiwoz/res.json', 'r') as f:
            res = json.load(f)
        
        for d, r in zip(datas, res):
            for k, v in d.items():
                if isinstance(v, str) and v.startswith('<sos'):
                    d[k] = v[8:-8].strip()
            d['pointer'] = r['pointer']
        torch.save(datas, self.saved_text_file)
        
        b, s, i = eval.validation_metric(datas)
        return {'bleu': '{:.2f}'.format(b), 'success': '{:.2f}'.format(s), 'inform': '{:.2f}'.format(i), 'overall': '{:.2f}'.format(b + (s + i) / 2)}   

    def eval(self, data):
        if self.config['dataset'] == 'multiwoz':
            self.model.eval()
            with torch.no_grad():
                res = self.multiwoz_eval()
            self.model.train()
            return res
        
        generate_corpus = []
        self.model.eval()
        with torch.no_grad():
            for batch_data in tqdm(data, ncols=80):
                generate_corpus.extend(self.model.generate(batch_data, data))
        self.model.train()
        reference_corpus = data.get_reference()
        with open(self.saved_text_file, 'w') as f:
            for gen in generate_corpus:
                f.write(' '.join(gen) + '\n')
        result = self.evaluator.evaluate(generate_corpus, reference_corpus)
        return result

    def fit(self, train_data, valid_data=None, test_data=None, verbose=True, saved=True):
        r"""Train the model based on the train data and the valid data.

        Args:
            train_data (DataLoader): the train data
            valid_data (DataLoader, optional): the valid data, default: None.
                                               If it's None, the early_stopping is invalid.
            verbose (bool, optional): whether to write training and evaluation information to logger, default: True
            saved (bool, optional): whether to save the model parameters, default: True

        Returns:
             (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        if self.start_epoch >= self.epochs or self.epochs <= 0:
            self._save_checkpoint(-1)

        for epoch_idx in range(self.start_epoch, self.epochs):
            # train
            torch.cuda.empty_cache()
            self.epoch_idx = epoch_idx
            training_start_time = time()
            train_loss = self._train_epoch(train_data, epoch_idx)
            training_end_time = time()
            train_loss = sum(train_loss) if isinstance(train_loss, tuple) else train_loss
            self.train_loss_dict[epoch_idx] = train_loss
            train_loss_output = \
                self._generate_train_loss_output(epoch_idx, training_start_time, training_end_time, train_loss)
            self.logger.info(train_loss_output)

            # eval
            if self.eval_step <= 0 or not valid_data:
                self._save_checkpoint(epoch_idx, suff=str(epoch_idx))
                update_output = 'Saving current: %s' % self.saved_model_file
                self.logger.info(update_output)
                continue

            if (epoch_idx + 1) % self.eval_step == 0:
                valid_start_time = time()
                test_score = self.eval(valid_data)
                valid_end_time = time()
                self.score_output = "epoch %d valid evaluating [time: %.2fs]\n%s" % (epoch_idx, valid_end_time - valid_start_time, test_score)
                self.logger.info(self.score_output)
                valid_start_time = time()
                test_score = self.eval(test_data)
                valid_end_time = time()
                self.score_output = "epoch %d test evaluating [time: %.2fs]\n%s" % (epoch_idx, valid_end_time - valid_start_time, test_score)
                self.logger.info(self.score_output)

                score = 0
                for metric in self.eval_metrics:
                    score += float(test_score[metric])
                if score > self.best_score:
                    self.cur_step = 0
                    self.best_score = score
                    self.best_result = self.score_output
                    self._save_checkpoint(epoch_idx)
                    update_output = 'Saving current best: %s' % self.saved_model_file
                    os.system(f"cp {self.saved_text_file} {self.saved_text_file + 'best'}")
                    self.logger.info(update_output)
                else:
                    self.cur_step += 1
                    if self.cur_step > self.stopping_step:
                        stop_output = 'Finished training, best eval result in epoch %d' % \
                                (epoch_idx - self.cur_step * self.eval_step)
                        self.logger.info(stop_output)
                        break
        return self.best_result

    @torch.no_grad()
    def evaluate(self, eval_data, load_best_model=True, model_file=None, eval=True):
        r"""Evaluate the model based on the eval data.

        Args:
            eval_data (DataLoader): the eval data
            load_best_model (bool, optional): whether load the best model in the training process, default: True.
                                              It should be set True, if users want to test the model after training.
            model_file (str, optional): the saved model file, default: None. If users want to test the previously
                                        trained model file, they can set this parameter.

        Returns:
            dict: eval result, key is the eval metric and value in the corresponding metric value
        """
        if model_file:
            checkpoint_file = model_file
        else:
            checkpoint_file = self.saved_model_file

        checkpoint = torch.load(checkpoint_file)
        self.model.load_state_dict(checkpoint['state_dict'])
        message_output = 'Loading model structure and parameters from {}'.format(checkpoint_file)
        self.logger.info(message_output)
        return self.eval(eval_data)

class Seq2SeqTrainer(Trainer):
    r"""Seq2SeqTrainer is designed for seq2seq testing, which is a typically used setting.
    """

    def __init__(self, config, model, batch_num):
        super(Seq2SeqTrainer, self).__init__(config, model, batch_num)

